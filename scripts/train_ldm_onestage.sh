#cd /data/lsr/code/style3d_gen
#export PYTHONPATH=/data/lsr/code/style3d_gen
#python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
#    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
#    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_multitype/encoder_mode \
#    --padding zero \
#    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond_multitype --train_nepoch 300000 --test_nepoch 1000 --save_nepoch 5000 \
#    --batch_size 1230 --max_face 32 --bbox_scaled 1.0 \
#    --pointcloud_encoder POINT_E --pointcloud_feature_dir /data/AIGP/pc_cond_sample_multitype \
#    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature --gpu 0
#
## Z Bsize 188:1230
#cd /data/lsr/code/style3d_gen
#export PYTHONPATH=/data/lsr/code/style3d_gen
#python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
#    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
#    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
#    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_multitype/encoder_mode \
#    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond_multitype --train_nepoch 300000 --test_nepoch 200 --save_nepoch 5000 \
#    --batch_size 2460 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
#    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
#    --pointcloud_encoder POINT_E --pointcloud_feature_dir /data/AIGP/pc_cond_sample_multitype \
#    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature

# 多种点云条件(还没训练，试试新的VAE)
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_ snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz_onestage \
    --denoiser_type default --scheduler DDPM --lr 5e-4\
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/stylexdQ1Q2Q4_surfzOnestage_xyzuv_mask_sketchCond_radio_v2.5-h_latent_8_8_1_vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfzOnestage_L12_emb768_dropout0.1_xyzuv_mask_pad_zero_sketchCond_radio_v2.5-h_latent_8_8_1_scheduler_DDPM \
    --train_nepoch 600000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 12 --pos_dim -1 --dropout 0.1 \
    --pointcloud_encoder POINT_E --pointcloud_feature_dir /data/AIGP/pc_cond_sample_multitype \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature \
    --gpu 0

# 草图条件RadioV.5h
# 188:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_ snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz_onestage \
    --denoiser_type default --scheduler DDPM --lr 5e-4\
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/stylexdQ1Q2Q4_surfzOnestage_xyzuv_mask_sketchCond_radio_v2.5-h_latent_8_8_1_vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfzOnestage_L12_emb768_dropout0.1_xyzuv_mask_pad_zero_sketchCond_radio_v2.5-h_latent_8_8_1_scheduler_DDPM \
    --train_nepoch 600000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 12 --pos_dim -1 --dropout 0.1 \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature \
    --gpu 0


# HY DDPM 测试添加Dropout的HYDIT的一阶段训练 Latent64(ncs+uv+M)+3DBBox+2DScale ===
# 【Loss震荡非常厉害】
# 190:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz_onestage \
    --denoiser_type hunyuan_dit --scheduler DDPM --lr 5e-4\
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/stylexdQ1Q2Q4_surfzOnestage_xyzuv_mask_sketchCond_radio_v2.5-h_latent_8_8_1_vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfzOnestage_HYdit_Layer_7_13_emb768_dropout0.1_xyzuv_mask_pad_zero_sketchCond_radio_v2.5-h_latent_8_8_1_scheduler_DDPM \
    --train_nepoch 600000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 7 13 --pos_dim -1 --dropout 0.1 \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
