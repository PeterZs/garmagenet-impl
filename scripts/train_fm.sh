


# FlowMatching latent64 vae0800  ===
# 6+8 latent64 pcCond_uniform vae0800
# 正在做 uniform sampling 的实验，做点云拆版，正好训一版FM对比LDM
# 188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python3 src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --device cuda --lr 5e-4\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/stylexdQ1Q2Q4_surfz_HYdit_Layer_6_8_emb768_xyzuv_pad_zero_pcCond_uniform_latent_8_8_1_scheduler_HY_FMED_shift1_lr5e-4_vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_6_8_emb768_xyzuv_pad_zero_pcCond_uniform_latent_8_8_1_scheduler_HY_FMED_shift1_lr5e-4_vae_e0800 --train_nepoch 300000 --test_nepoch 50 --save_nepoch 1000 \
    --batch_size 1230 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 6 8 \
    --scheduler HY_FMED --scheduler_shift 1 \
    --pointcloud_encoder POINT_E --pointcloud_sampled_dir /data/AIGP/pc_cond_sample/uniform_2048 \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature



# FlowMatching wcs ===
# 7+13 Latent256 vae_e0450.pt
# 188【还没跑】目的是为了给定版自动建模，但是仅仅这样没法对板片形状约束
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --lr 5e-5\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/ckpts/vae_e0450.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/cache/stylexdQ1Q2Q4_surfz_HYdit_Layer_7_13_emb768_xyz-w_normal_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift1_lr5e-5_vae_e0450/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_7_13_emb768_xyz-w_normal_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift1_lr5e-5_vae_e0450 --train_nepoch 100000 --test_nepoch 50 --save_nepoch 1000 \
    --batch_size 615 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 7 13 \
    --scheduler HY_FMED --scheduler_shift 1 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_wcs surf_normals surf_mask surf_uv_bbox_wcs sketch_feature


# 5+15 latent256
# 187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --device cuda --lr 5e-5\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/stylexdQ1Q2Q4_surfz_HYdit_Layer_5_15_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift5_lr5e-5/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_5_15_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift5_lr5e-5 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 500 \
    --batch_size 400 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 5 15 \
    --scheduler HY_FMED --scheduler_shift 1 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature



# 用于测试为啥 segment fault ===
# 187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --device cuda \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_stylexdQ1Q2Q4_surfz_HYdit_Layer_10_12_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift3/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_10_12_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift3 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 500 \
    --batch_size 100 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 15 30 \
    --scheduler HY_FMED --scheduler_shift 3 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
