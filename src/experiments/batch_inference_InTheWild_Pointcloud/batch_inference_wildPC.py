"""
仅用于SiggrapphAsia2025，后续仅供参考，不在此基础上继续编辑
"""

import os
import pickle
import argparse
from tqdm import tqdm
from glob import glob

import torch
import trimesh
import numpy as np
from diffusers import DDPMScheduler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go

from src.utils import randn_tensor
from src.pc_utils import normalize_pointcloud
from src.vis import draw_bbox_geometry
from src.network import AutoencoderKLFastDecode, SurfZNet, SurfPosNet, TextEncoder, PointcloudEncoder
from src.bbox_utils import bbox_deduplicate
from src.pc_utils import farthest_point_sample

def pointcloud_condition_visualize(vertices: np.ndarray, output_fp=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices should be ndarray in (Nx3)"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False
            ),
            showlegend=False
        )
    ])

    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=3)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 800
    if output_fp:
        # fig.write_html(output_fp.replace(".pkl", "") + "_pcCond_vis.html")
        fig.write_image(output_fp, width=RESO, height=RESO, scale=2.5)



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

    surf_vae = AutoencoderKLFastDecode( in_channels=6,
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
    surf_vae.load_state_dict(torch.load(args.vae, map_location="cuda"), strict=False)
    surf_vae.to('cuda').eval()

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    # Load conditioning model
    if args.text_encoder is not None: text_enc = TextEncoder(args.text_encoder, device='cuda')
    else: text_enc = None

    if args.text_encoder is not None:
        condition_dim = text_enc.text_emb_dim
    elif args.pointcloud_encoder is not None:
        condition_dim = 512
    elif args.sketch_encoder is not None:
        if args.sketch_encoder == "LAION2B":
            condition_dim = 1280
        else:
            raise NotImplementedError("args.sketch_encoder name wrong.")
    else:
        condition_dim = -1

    # Load SurfPos Net
    surfpos_model = SurfPosNet(
        p_dim=10,
        condition_dim=condition_dim,
        num_cf=-1
    )
    surfpos_model.load_state_dict(torch.load(args.surfpos)['model_state_dict'])
    surfpos_model.to('cuda').eval()
    # Load SurfZ Net
    surfz_model = SurfZNet(
        p_dim=10,
        z_dim=latent_size**2*latent_channels,
        num_heads=12,
        condition_dim=condition_dim,
        num_cf=-1
        )
    surfz_model.load_state_dict(torch.load(args.surfz)['model_state_dict'])
    surfz_model.to('cuda').eval()

    print('[DONE] Models initialized.')

    return {
        'surf_vae': surf_vae,
        'ddpm_scheduler': ddpm_scheduler,
        'surfpos_model': surfpos_model,
        'surfz_model': surfz_model,
        'text_enc': text_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size,
    }


def inference_one(
        models,
        args,
        caption='',
        pointcloud_feature=None,
        sketch_features=None,
        surf_pos_orig=None,
        output_fp='',
        dedup=True,
        vis=False,
        data_fp=None
):
    batch_size = 1
    device = 'cuda'
    max_surf = 32

    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surfpos_model = models['surfpos_model']
    surf_vae = models['surf_vae']
    text_enc = models['text_enc']

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    if args.text_encoder is not None and caption is not None:
        condition_emb = text_enc(caption)
    elif args.pointcloud_encoder is not None and pointcloud_feature is not None:
        condition_emb = torch.tensor(pointcloud_feature,device=device)
    elif args.sketch_encoder is not None and sketch_features is not None:
        condition_emb = sketch_features
    else:
        condition_emb = None

    # GET BBOX ---------------------------------------------------------------
    # Generate bbox
    if surf_pos_orig is None:
        surfPos = randn_tensor((batch_size, max_surf, 10)).to(device)
        ddpm_scheduler.set_timesteps(1000)
        with torch.no_grad():
            for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
                timesteps = t.reshape(-1).to(device)
                pred = surfpos_model(surfPos, timesteps, condition=condition_emb)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample

        if dedup:
            bboxes, dedup_mask = bbox_deduplicate(surfPos, padding=args.padding, dedup_repeat=True)
            surf_mask = torch.zeros((1, len(bboxes))) == 1
            _surf_pos = torch.concat([torch.FloatTensor(bboxes), torch.zeros(max_surf - len(bboxes), 10)]).unsqueeze(0)
            _surf_mask = torch.concat([surf_mask, torch.zeros(1, max_surf - len(bboxes)) == 0], -1)
            n_surfs = torch.sum(~_surf_mask)

    else:
        n_surfs = torch.tensor(len(surf_pos_orig))
        surf_mask = torch.zeros((1, n_surfs)) == 1
        _surf_pos = torch.concat([surf_pos_orig, torch.zeros((max_surf - n_surfs, 10))]).to(device)
        _surf_mask = torch.concat([surf_mask, torch.zeros(1, max_surf - n_surfs) == 0], -1).to(device)


    # SurfZ
    _surf_z = randn_tensor((1, max_surf, latent_channels * latent_size * latent_size), device='cuda')
    ddpm_scheduler.set_timesteps(1000)
    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Z Denoising"):
            timesteps = t.reshape(-1).to('cuda')
            pred = surfz_model(
                _surf_z, timesteps, _surf_pos.to('cuda'), _surf_mask.to('cuda'), condition_emb, is_train=False)
            _surf_z = ddpm_scheduler.step(pred, t, _surf_z,).prev_sample
    _surf_z = _surf_z[:,:n_surfs]

    # VAE Decoding
    with torch.no_grad(): decoded_surf_pos = surf_vae(_surf_z.view(-1, latent_channels, latent_size, latent_size))
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

        if output_fp: plt.savefig(output_fp.replace(os.path.splitext(output_fp)[-1], '_geo_img.png'), transparent=True, dpi=72)
        else: plt.show()
        plt.close()

    # plotly visualization
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_surf_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf_pos = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs_mask = _decoded_surf_pos[..., -1:].reshape(n_surfs, -1) > 0.0
    _surf_ncs = _decoded_surf_pos[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf_pos[..., 3:5].reshape(n_surfs, -1, 2)

    _surf_uv_bbox = _surf_bbox[..., 6:]
    _surf_bbox = _surf_bbox[..., :6]

    if vis:
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]
        _surf_wcs = _denormalize_pts(_surf_ncs, _surf_bbox)
        draw_bbox_geometry(
            bboxes = _surf_bbox,
            bbox_colors = colors,
            points = _surf_wcs,
            point_masks = _surf_ncs_mask,
            point_colors = colors,
            num_point_samples = 5000,
            title = caption,
            output_fp = output_fp.replace('.pkl', '_pointcloud.png'),
            # show_num=True
            )

    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption,
        "data_fp": data_fp
    }

    if output_fp:
        with open(output_fp, 'wb') as f: pickle.dump(result, f)


def run(args):
    # 保证仅采用一种 condition embedding
    not_none_count = sum(x is not None for x in [args.text_encoder, args.pointcloud_encoder, args.sketch_encoder])
    assert not_none_count in (0, 1)

    models = init_models(args)

    os.makedirs(args.output, exist_ok=True)
    pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, "cuda")

    reference_pc_dir = "/home/Ex1/data/resources/其它/2025_10_15_Hunyuan结果做点云生版/2_论文中的DF点云渲染图做混元3D生成/PC_Mesh"

    suffixs = [".ply", ".obj", ".glb"]
    reference_pc_fp_list = []
    for suffix in suffixs:
        reference_pc_fp_list.extend(sorted(glob(os.path.join(reference_pc_dir, "**", f"*{suffix}"), recursive=True)))

    for data_fp in tqdm(reference_pc_fp_list):
        mesh = trimesh.load(data_fp, process=False, force="mesh")
        pc = np.array(mesh.vertices)

        suffix = os.path.splitext(data_fp)[-1]
        output_fp = os.path.join(args.output, f'{os.path.basename(data_fp).replace(suffix, ".pkl")}')

        if args.pc_sample_type == "random":
            sampled_pts = pc[np.random.randint(0, len(pc), size=2048)] *3  # pc[np.random.choice(np.arange(len(pc)), size=2048)]
        elif args.pc_sample_type == "fps":
            sampled_pts = farthest_point_sample(pc ,2048, 40000)[0]
        else:
            raise NotImplementedError

        sampled_pts_normalized = normalize_pointcloud(sampled_pts, 1)
        pointcloud_condition_visualize(sampled_pts_normalized, output_fp.replace(".pkl","")+"_pcCond_orig.png")
        np.save(output_fp.replace(".pkl","")+"_pcCond_orig.npy", sampled_pts)

        pointcloud_features = pointcloud_encoder(sampled_pts)
        inference_one(
            models,
            args=args,
            caption = None,
            pointcloud_feature=pointcloud_features,
            sketch_features=None,
            surf_pos_orig = None,
            dedup=True,
            output_fp = output_fp,
            vis=True,
            data_fp = None
        )
    print('[DONE]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument(
        '--vae', type=str,
        default='/data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt',
        help='Path to VAE model')
    parser.add_argument(
        '--surfpos', type=str,
        default='/data/lsr/models/style3d_gen/surf_pos/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--surfz', type=str,
        default='/data/lsr/models/style3d_gen/surf_z/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e200000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--cache', type=str,
        default='/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode/surfpos_validate.pkl',
        help='Path to cache file')
    parser.add_argument('--use_original_pos', type=bool,
                        default=False, help='If using pos information in training/validating data.')
    parser.add_argument(
        '--output', type=str, default='generated/20250818_revision_wildPC',
        help='Path to output directory')
    parser.add_argument(
        '--pc_sample_type', type=str,choices=["random", "fps"], default='random',
        help='')
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--reso', type=int, default=256, help='Sample size of the vae model.')
    parser.add_argument("--padding", type=str, default="zero", choices=["repeat", "zero"], help='Padding type during surfPos training.')
    parser.add_argument('--text_encoder', type=str, default=None, help='Text encoder type.')
    parser.add_argument('--pointcloud_encoder', type=str, default="POINT_E", help='Text encoder type.')
    parser.add_argument('--sketch_encoder', type=str, default=None, help='Text encoder type.')
    args = parser.parse_args()
    run(args)