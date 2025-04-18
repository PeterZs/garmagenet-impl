import os
import random
from glob import glob
import argparse
import pickle
import numpy as np

from tqdm import tqdm

import torch
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from src.network import AutoencoderKLFastDecode, AutoencoderKLFastEncode, SurfZNet, SurfPosNet, TextEncoder
from diffusers import DDPMScheduler  # , PNDMScheduler
from src.utils import randn_tensor
from src.vis import draw_bbox_geometry, draw_bbox_geometry_3D2D


def _denormalize_pts(pts, bbox):    
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def init_models(args):
        
    block_dims = args.block_dims
    latent_channels = args.latent_channels
    sample_size = args.reso
    latent_size = sample_size//(2**(len(block_dims)-1))
    
    surf_vae_decoder = AutoencoderKLFastDecode( in_channels=6,
                                    out_channels=6,
                                    down_block_types=['DownEncoderBlock2D']*len(block_dims),
                                    up_block_types=['UpDecoderBlock2D']*len(block_dims),
                                    block_out_channels=block_dims,
                                    layers_per_block=2,
                                    act_fn='silu',
                                    latent_channels=latent_channels,
                                    norm_num_groups=8,
                                    sample_size=sample_size
                                    )
    surf_vae_decoder.load_state_dict(torch.load(args.vae), strict=False)
    surf_vae_decoder.to('cuda').eval()

    surf_vae_encoder = AutoencoderKLFastEncode(
        in_channels=6,
        out_channels=6,
        down_block_types=['DownEncoderBlock2D']*len(block_dims),
        up_block_types=['UpDecoderBlock2D']*len(block_dims),
        block_out_channels=block_dims,
        layers_per_block=2,
        act_fn='silu',
        latent_channels=latent_channels,
        norm_num_groups=8,
        sample_size=sample_size,
    )
    surf_vae_encoder.load_state_dict(torch.load(args.vae), strict=False)
    surf_vae_encoder.to('cuda').eval()

    # pndm_scheduler = PNDMScheduler(
    #     num_train_timesteps=1000,
    #     beta_schedule='linear',
    #     prediction_type='epsilon',
    #     beta_start=0.0001,
    #     beta_end=0.02,
    # )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    # Load conditioning model
    if "text_encoder" in args: text_enc = TextEncoder(args.text_encoder, device='cuda')
    else: text_enc = None

    # Load SurfPos Net
    surfpos_model = SurfPosNet(
        p_dim=10,
        condition_dim=text_enc.text_emb_dim if "text_encoder" in args else -1,
        num_cf=-1
    )
    surfpos_model.load_state_dict(torch.load(args.surfpos)['model_state_dict'])
    surfpos_model.to('cuda').eval()
    # Load SurfZ Net
    surfz_model = SurfZNet(
        p_dim=10, 
        z_dim=latent_size**2*latent_channels, 
        num_heads=12, 
        condition_dim=text_enc.text_emb_dim if "text_encoder" in args else -1,
        num_cf=-1
        )
    surfz_model.load_state_dict(torch.load(args.surfz)['model_state_dict'])
    surfz_model.to('cuda').eval()
    
    print('[DONE] Models initialized.')

    return {
        'surf_vae_decoder': surf_vae_decoder,
        'surf_vae_encoder': surf_vae_encoder,
        'ddpm_scheduler': ddpm_scheduler,
        'surfpos_model': surfpos_model,
        'surfz_model': surfz_model,
        'text_enc': text_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size
    }


def inference_one(models,
                  surf_wcs,
                  surf_uv_wcs,
                  surf_ncs,
                  surf_uv_ncs,
                  surf_mask,
                  surf_bbox_wcs,
                  surf_uv_bbox_wcs,
                  bbox_mask,
                  caption='',
                  output_dir='',
                  data_id=0,
                  dedup=False,
                  vis=False,
                  export_vid=False):
    batch_size = 1
    device = 'cuda'
    max_surf = 32
    bbox_dim = 10

    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surfpos_model = models['surfpos_model']
    surf_vae_decoder = models['surf_vae_decoder']
    surf_vae_encoder = models['surf_vae_encoder']
    text_enc = models.get('text_enc', None)

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    # SurfPos ===
    surf_pos_gt = torch.tensor(np.concatenate([surf_bbox_wcs, surf_uv_bbox_wcs], axis=-1), device=device)
    n_surfs, n_pads = surf_pos_gt.shape[0], max_surf-surf_pos_gt.shape[0]

    # pad zero
    pad_idx = torch.arange(n_surfs)
    _panel_mask = torch.cat([
        torch.zeros(n_surfs, dtype=bool), torch.ones(n_pads, dtype=bool)
    ], dim=0)[None, ...]
    _surf_pos_gt = torch.cat([
        surf_pos_gt[pad_idx, ...], torch.zeros((n_pads, *surf_pos_gt.shape[1:]), dtype=surf_pos_gt.dtype, device=surf_pos_gt.device)
    ], dim=0)[None, ...]

    # text embedding
    text_embeding = text_enc(caption) \
        if caption is not None and text_enc is not None else None


    # Mask Choice
    # 1.仅重新生成部分3D或2DBBox
    # 2.全部2D重新生成
    # 3.全部重新生成
    masktype = 1
    if masktype==1:
        # per-dim mask of BBox
        _surf_pos_mask = torch.ones((batch_size, max_surf, bbox_dim), dtype=torch.bool, device=device)
        for idx, m in enumerate(bbox_mask):
            if not m: _surf_pos_mask[0, idx]=False

        # impainting操作（可以指定3D或2DBBox不变，仅恢复另一种BBox）
        _surf_pos_mask[:, :, :6] = True  # only edit 2D
        # _surf_pos_mask[:, :, 6:] = False  # only edit 3D
    elif masktype==2:
        _surf_pos_mask = torch.zeros((batch_size, max_surf, bbox_dim), dtype=torch.bool, device=device)
        _surf_pos_mask[:,:,:6] = True
    elif masktype==3:
        _surf_pos_mask = torch.zeros((batch_size, max_surf, bbox_dim), dtype=torch.bool, device=device)
    else:
        raise ValueError

    # generate bbox by denoising
    _surf_pos = randn_tensor((batch_size, max_surf, bbox_dim)).to(device)
    ddpm_scheduler.set_timesteps(1000)
    if export_vid:
        ROOT = "generated/surfpos_denoising"
        ID = len(glob(os.path.join(ROOT, "*")))
        DIR = os.path.join(ROOT, f"{ID}".zfill(3))
        os.makedirs(DIR, exist_ok=True)
    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
            _surf_pos_gt_noise = torch.randn(_surf_pos.shape).to(device)
            _surf_pos_gt_noised = ddpm_scheduler.add_noise(_surf_pos_gt, _surf_pos_gt_noise, t)

            #
            _surf_pos[_surf_pos_mask] = _surf_pos_gt_noised[_surf_pos_mask]

            timesteps = t.reshape(-1).to(device)
            pred = surfpos_model(_surf_pos, timesteps, condition=text_embeding)
            _surf_pos = ddpm_scheduler.step(pred, t, _surf_pos).prev_sample
            # # [test]
            # if export_vid and (t%20==0 or t>999 or t<30):
            #     FP = os.path.join(DIR,f"{t}".zfill(3)+".png")
            #     draw_bbox_geometry(
            #         _surf_pos[0][:,:6].detach().cpu().numpy(),
            #         [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, 32)],
            #         output_fp=FP
            #     )
    _surf_pos[_surf_pos_mask] = _surf_pos_gt[_surf_pos_mask]

    colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
    _surf_pos_ = _surf_pos[:, :n_surfs]
    _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    _surf_uv_bbox_wcs_[:,[0,1,3,4]] = _surf_pos_[0,:,6:].detach().cpu().numpy()

    # # OLD ===
    # _surf_pos_mask = torch.ones((batch_size, max_surf, bbox_dim), dtype=torch.bool, device=device)
    # for idx, m in enumerate(bbox_mask):
    #     if not m: _surf_pos_mask[0, idx] = False
    # # impainting操作（可以指定3D或2DBBox不变，仅恢复另一种BBox）
    # _surf_pos_mask[:, :, :6] = False  # only edit 2D
    # # _surf_pos_mask[:, :, 6:] = False  # only edit 3D
    #
    # # generate bbox by denoising
    # _surf_pos = randn_tensor((batch_size, max_surf, bbox_dim)).to(device)
    # ddpm_scheduler.set_timesteps(1000)
    # if export_vid:
    #     ROOT = "generated/surfpos_denoising"
    #     ID = len(glob(os.path.join(ROOT, "*")))
    #     DIR = os.path.join(ROOT, f"{ID}".zfill(3))
    #     os.makedirs(DIR, exist_ok=True)
    # with torch.no_grad():
    #     for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
    #         _surf_pos_gt_noise = torch.randn(_surf_pos.shape).to(device)
    #         _surf_pos_gt_noised = ddpm_scheduler.add_noise(_surf_pos_gt, _surf_pos_gt_noise, t)
    #
    #         _surf_pos[~_surf_pos_mask] = _surf_pos_gt_noised[~_surf_pos_mask]
    #
    #         timesteps = t.reshape(-1).to(device)
    #         pred = surfpos_model(_surf_pos, timesteps, condition=text_embeding)
    #         _surf_pos = ddpm_scheduler.step(pred, t, _surf_pos).prev_sample
    #         # # [test]
    #         # if export_vid and (t%20==0 or t>999 or t<30):
    #         #     FP = os.path.join(DIR,f"{t}".zfill(3)+".png")
    #         #     draw_bbox_geometry(
    #         #         _surf_pos[0][:,:6].detach().cpu().numpy(),
    #         #         [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, 32)],
    #         #         output_fp=FP
    #         #     )
    # _surf_pos[_surf_pos_mask] = _surf_pos_gt[_surf_pos_mask]
    # colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
    # _surf_pos_ = _surf_pos[:, :n_surfs]
    # _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    # _surf_uv_bbox_wcs_[:,[0,1,3,4]] = _surf_pos_[0,:,6:].detach().cpu().numpy()



    #vis
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0,:,:6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    # text_embeding = text_enc(caption) if caption and text_enc is not None else None

    # Diffusion Generation
    surf_gt = torch.tensor(np.concatenate([surf_ncs, surf_uv_ncs, surf_mask*2.-1], axis=-1), device=device, dtype=torch.float32)
    # pad zero
    _surf_gt = torch.cat([
        surf_gt[pad_idx, ...], torch.zeros((n_pads, *surf_gt.shape[1:]), dtype=surf_gt.dtype, device=surf_gt.device)
    ], dim=0)[None, ...]

    with torch.no_grad():
        _surf_z_gt = (surf_vae_encoder(_surf_gt[0].permute(0,3,1,2))
                              ).reshape(1, 32, latent_channels * latent_size * latent_size)

    _surf_z_mask = torch.tensor(np.concatenate([bbox_mask,np.ones(n_pads,dtype=np.bool)]), device=device)
    _surf_z = randn_tensor((1, 32, latent_channels * latent_size * latent_size), device='cuda')
    ddpm_scheduler.set_timesteps(1000)
    if export_vid:
        ROOT = "generated/surfz_denoising"
        ID = len(glob(os.path.join(ROOT, "*")))
        DIR = os.path.join(ROOT, f"{ID}".zfill(3))
        os.makedirs(DIR, exist_ok=True)
    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Z Denoising"):
            _surf_z_gt_noise = torch.randn(_surf_z.shape).to(device)
            _surf_z_gt_noised = ddpm_scheduler.add_noise(_surf_z_gt, _surf_z_gt_noise, t)

            _surf_z[0][_surf_z_mask] = _surf_z_gt_noised[0][_surf_z_mask]

            timesteps = t.reshape(-1).to('cuda')
            pred = surfz_model(_surf_z, timesteps, _surf_pos.to('cuda'),
                               _panel_mask.to('cuda'), class_label=None, condition=text_embeding, is_train=False)
            _surf_z = ddpm_scheduler.step(pred, t, _surf_z).prev_sample

            if export_vid and (t%200==0 or t>998 or t<2):
                colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
                _surf_z_ = _surf_z[:, :n_surfs]
                with torch.no_grad(): decoded_surf_z = surf_vae_decoder(_surf_z_.view(-1, latent_channels, latent_size, latent_size))
                _surf_bbox_ = _surf_pos.squeeze(0)[~_panel_mask.squeeze(0), :].detach().cpu().numpy()
                _decoded_surf_z_ = decoded_surf_z.permute(0, 2, 3, 1).detach().cpu().numpy()
                _surf_ncs_mask_ = _decoded_surf_z_[..., -1:].reshape(n_surfs, -1) > 0.0
                _surf_ncs_ = _decoded_surf_z_[..., :3].reshape(n_surfs, -1, 3)
                _surf_bbox_ = _surf_bbox_[..., :6]
                _surf_wcs_ = _denormalize_pts(_surf_ncs_, _surf_bbox_)

                FP = os.path.join(DIR,f"{t}".zfill(3)+".png")
                draw_bbox_geometry(
                    bboxes = _surf_bbox_,
                    bbox_colors = colors,
                    points = _surf_wcs_,
                    point_colors = colors,
                    point_masks = _surf_ncs_mask_,
                    num_point_samples = 5000,
                    title = caption,
                    output_fp=FP,
                )

    surf_z = _surf_z[:,:n_surfs]

    #vis
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0,:,:6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    # VAE Decoding
    with torch.no_grad(): decoded_surf_pos = surf_vae_decoder(surf_z.view(-1, latent_channels, latent_size, latent_size))
    pred_img = make_grid(decoded_surf_pos, nrow=6, normalize=True, value_range=(-1,1))

    if vis:
        fig, ax = plt.subplots(3, 1, figsize=(40, 40))
        ax[0].imshow(pred_img[:3, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[1].imshow(pred_img[3:, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[2].imshow(pred_img[-1:, ...].permute(1, 2, 0).detach().cpu().numpy())

        ax[0].set_title('Geometry Images')
        ax[1].set_title('UV Images')
        ax[2].set_title('Mask Images')

        plt.tight_layout()
        plt.axis('off')

        if output_dir: plt.savefig(os.path.join(output_dir, f"{data_id}".zfill(5)+"_geo_img.png"), transparent=True, dpi=72)
        else: plt.show()
        plt.close()

    # plotly visualization
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_panel_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf_pos = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs_mask = _decoded_surf_pos[..., -1:].reshape(n_surfs, -1) > 0.0
    _surf_ncs = _decoded_surf_pos[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf_pos[..., 3:5].reshape(n_surfs, -1, 2)

    _surf_uv_bbox = _surf_bbox[..., 6:]
    _surf_bbox = _surf_bbox[..., :6]

    if vis:
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]
        _surf_wcs = _denormalize_pts(_surf_ncs, _surf_bbox)
        # _surf_uv_wcs = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox)
        draw_bbox_geometry( # [todo] 看
            bboxes = _surf_bbox,
            bbox_colors = colors,
            points = _surf_wcs,
            point_masks = _surf_ncs_mask,
            point_colors = colors,
            num_point_samples = 5000,
            title = caption,
            output_fp =  plt.savefig(os.path.join(output_dir, f"{data_id}".zfill(5)+"_pointcloud.png"))
            # show_num=True
            )

    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption              # str
    }

    if output_dir:
        with open(os.path.join(output_dir, f"{data_id}".zfill(5)+".pkl"), 'wb') as f: pickle.dump(result, f)
    print('[DONE] save to:', output_dir)


def run(args):
    models = init_models(args)
    
    os.makedirs(args.output, exist_ok=True)

    data_list = sorted(glob(os.path.join(args.data_root,"*")))

    for data_dir in data_list:
        data_path = glob(os.path.join(data_dir,args.exp,"*.pkl"))[0]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        a=1
        sample_data_idx = int(data_dir.split('/')[-1])
        surf_wcs = data["surf_wcs"]
        surf_uv_wcs = data["surf_uv_wcs"]
        surf_ncs = data["surf_ncs"]
        surf_uv_ncs = data["surf_uv_ncs"]
        surf_mask = data["surf_mask"]
        surf_bbox_wcs = data["surf_bbox_wcs"]
        surf_uv_bbox_wcs = data["surf_uv_bbox_wcs"]
        bbox_mask = data["bbox_mask"]

        surf_cls = None
        caption = data["caption"]
        output_dir = os.path.join(args.output, f'{sample_data_idx}'.zfill(5))
        os.makedirs(output_dir, exist_ok=True)
        inference_one(models,
                      surf_wcs,
                      surf_uv_wcs,
                      surf_ncs,
                      surf_uv_ncs,
                      surf_mask,
                      surf_bbox_wcs,
                      surf_uv_bbox_wcs,
                      bbox_mask,
                      caption, output_dir, sample_data_idx, vis=True, export_vid=True)
    # for sample_data_idx in tqdm(sample_idxs):

    #
    #     output_fp = os.path.join(args.output, f'{sample_data_idx:04d}.pkl')
    #
    #     inference_one(models, surf_pos, surf_cls, caption, output_fp, vis=True, export_vid=False)
    #     if sample_data_idx==4:
    #         break
    # print('[DONE]')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp', type=str,
        default='bbox_seg',
        help='Experiment type')

    # path configuration
    parser.add_argument(
        '--vae', type=str, 
        default='/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e550.pt',
        help='Path to VAE model')
    parser.add_argument(
        '--surfpos', type=str,
        default='/home/Ex1/data/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_repeat_cond_clip/ckpts/surfpos_e60000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--surfz', type=str,
        default='/home/Ex1/data/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode_with_caption/ckpts/surfz_e100000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--text_encoder', type=str,
        default='CLIP',
        help='Path to SurfZ model')
    parser.add_argument(
        '--data_root', type=str,
        default='_LSR/gen_experiment_data/bbox_editing/output',
        help='Path to cache file')
    parser.add_argument(
        '--output', type=str, default='generated/surfz_e230000_bbox_editing',
        help='Path to output directory')
    
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--reso', type=int, default=256, help='Sample size of the vae model.')
    
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to inference.')

    args = parser.parse_args()
    
    run(args)