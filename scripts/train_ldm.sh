#!/bin/bash\

### Train the Latent Diffusion Model ###
# --data_aug is optional
# max_face 30, max_edge 20 for deepcad 
# max_face 50, max_edge 30 for abc/furniture
# --surfvae refer to the surface vae weights 
# --edgevae refer to the edge vae weights

# 256 lsr ===
# zero padding xyz uv mask Q1-2 pcCond ===
# POS  Bsize 199:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --padding zero \
    --expr stylexd_surfpos_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 3420 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
# Z   Bsize 199:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --expr stylexd_surfz_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 3420 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature

# zero padding xyz uv mask Q1-4 sketchCond ===
# POS Bsize 187:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 2500 --max_face 32 --bbox_scaled 1.0 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
# Z Bsize 188:3420 190:5000
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 5000 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature


# zero padding xyz uv mask Q1-4 pcCond ===
# POS Bsize 187:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 3420 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
# Z Bsize 188:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 3420 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature


# zero padding xyz uv mask Q1-4 unCond
# POS
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs
# Z
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs
# Z resume training
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
     --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz \
     --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
     --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
     --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
     --batch_size 5000 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
     --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
     --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs \
     --finetune --weight /data/lsr/code/style3d_gen/log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond/ckpts/surfz_e10000.pt
# --text_encoder CLIP


# 256 lry ===
### StyleXD - SurfPos
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat \
    --expr stylexd_surfpos_xyzuv_pad_repeat_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs


python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat --text_encoder CLIP \
    --expr stylexd_surfpos_xyzuv_pad_repeat_cond_clip_debug_1 --train_nepoch 5 --test_nepoch 2 --save_nepoch 3 \
    --batch_size 4 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs caption


### StyleXD - SurfZ
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption


python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption_test --train_nepoch 10 --test_nepoch 5 --save_nepoch 10 \
    --batch_size 1024 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption


CUDA_VISIBLE_DEVICES=0 python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e300.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e300/encoder_mode \
    --expr stylexd_surfz_xyzuv_latent1_mode --train_nepoch 50000 --test_nepoch 50 --save_nepoch 500 \
    --batch_size 1024 --chunksize -1 --padding zero --bbox_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs 


### Training DeepCAD Latent Diffusion Model ###  
python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option surfz \
    --surfvae log/deepcad_vae_surf.pt --gpu 0 1 \
    --expr deepcad_ldm_surfz --train_nepoch 3000 --batch_size 256 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option edgepos \
    --surfvae log/deepcad_vae_surf.pt --gpu 0 1 \
    --expr deepcad_ldm_edgepos --train_nepoch 1000 --batch_size 128 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option edgez \
    --surfvae log/deepcad_vae_surf.pt --edgevae log/deepcad_vae_edge.pt --gpu 0 1 \
    --expr deepcad_ldm_edgez --train_nepoch 1000 --batch_size 128 \
    --max_face 30 --max_edge 20


### Training ABC Latent Diffusion Model ###  
python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr abc_ldm_surfpos --train_nepoch 1000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option surfz \
    --surfvae log/abc_vae_surf.pt --gpu 0 1 \
    --expr abc_ldm_surfz --train_nepoch 1000 --batch_size 256 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option edgepos \
    --surfvae log/abc_vae_surf.pt --gpu 0 1 \
    --expr abc_ldm_edgepos --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option edgez \
    --surfvae log/abc_vae_surf.pt --edgevae log/abc_vae_edge.pt --gpu 0 1 \
    --expr abc_ldm_edgez --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30


### Training Furniture Latent Diffusion Model (classifier-free) ###  
python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr furniture_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option surfz \
    --surfvae log/furniture_vae_surf.pt --gpu 0 1 \
    --expr furniture_ldm_surfz --train_nepoch 3000 --batch_size 256 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option edgepos \
    --surfvae log/furniture_vae_surf.pt --gpu 0 1 \
    --expr furniture_ldm_edgepos --train_nepoch 1000 --batch_size 64 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option edgez \
    --surfvae log/furniture_vae_surf.pt --edgevae log/furniture_vae_edge.pt --gpu 0 1 \
    --expr furniture_ldm_edgez --train_nepoch 1000 --batch_size 64 \
    --max_face 50 --max_edge 30 --use_cf