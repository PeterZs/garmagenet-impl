# L24 Latent256 default-model vae_e0850.pt
# 测试 default model 的效果
# 190
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --lr 5e-4\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type default \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/ckpts/vae_e0450.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/cache/stylexdQ1Q2Q4_surfz_xyz-w_normal_mask_sketchCond_radio_v2.5-h_latent_16_16_1_vae_e0450/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_L24_emb768_posdim-1_xyz-w_normal_mask_pad_zero_sketchCond_radio_v2.5-h_latent_16_16_1_scheduler_DDPM_lr5e-4_vae_e0450 \
    --train_nepoch 100000 --test_nepoch 50 --save_nepoch 5000 \
    --batch_size 1230 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 24 --pos_dim -1\
    --scheduler DDPM \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_wcs surf_normals surf_mask surf_uv_bbox_wcs sketch_feature




# 7+13 Latent256 HYdit vae_e0450.pt
# 测试 Dropout=0, compare
# 190
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --lr 5e-4\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/ckpts/vae_e0450.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/cache/stylexdQ1Q2Q4_surfz_xyz-w_normal_mask_sketchCond_radio_v2.5-h_latent_16_16_1_vae_e0450/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_7_13_emb768_dropout0_posdim-1_xyz-w_normal_mask_pad_zero_sketchCond_radio_v2.5-h_latent_16_16_1_scheduler_DDPM_lr5e-4_vae_e0450 \
    --train_nepoch 100000 --test_nepoch 50 --save_nepoch 5000 \
    --batch_size 1230 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 7 13 --pos_dim -1 --dropout 0.\
    --scheduler DDPM \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_wcs surf_normals surf_mask surf_uv_bbox_wcs sketch_feature




# 7+13 Latent256 vae_e0850.pt
# 测试增加Dropout后的泛化性
# 190
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --lr 5e-4\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/ckpts/vae_e0450.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1/cache/stylexdQ1Q2Q4_surfz_xyz-w_normal_mask_sketchCond_radio_v2.5-h_latent_16_16_1_vae_e0450/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_7_13_emb768_dropout0.1_posdim-1_xyz-w_normal_mask_pad_zero_sketchCond_radio_v2.5-h_latent_16_16_1_scheduler_DDPM_lr5e-4_vae_e0450 \
    --train_nepoch 100000 --test_nepoch 50 --save_nepoch 5000 \
    --batch_size 1230 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 7 13 --pos_dim -1 --dropout 0.1\
    --scheduler DDPM \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_wcs surf_normals surf_mask surf_uv_bbox_wcs sketch_feature
