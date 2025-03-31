## Requirements

### Environment (Tested)
- Linux
- Python 3.9
- CUDA 11.8 
- PyTorch 2.2 
- Diffusers 0.27


### Dependencies

Install PyTorch and other dependencies:
```
conda create --name garmage_env python=3.9 -y
conda activate garmage_env

pip install -r requirements.txt
pip install chamferdist
```

If `chamferdist` fails to install here are a few options to try:

- If there is a CUDA version mismatch error, then try setting the `CUDA_HOME` environment variable to point to CUDA installation folder. The CUDA version of this folder must match with PyTorch's version i.e. 11.8.

- Try [building from source](https://github.com/krrish94/chamferdist?tab=readme-ov-file#building-from-source).


## Data
The dataset for training consists of `*.pkl` files for each garment (e.g. `resources/examples/processes/00000.pkl`) containing the following fields:
```python
result = {
    # raw data path
    'data_fp': data_item,
    # comma-separated description for the garmage
    'caption': "dress, fitted, round neck, puff sleeves, empire waist", 
    
    # (3, ), global offset for all garments default to [0., 1000., 0.]
    'global_offset': global_offset.astype(np.float32),  
    # float, global scale for all garments default to 2000
    'global_scale': global_scale,                 
    # (2, ) global uv offset, default to [0., 1000.]
    'uv_offset': uv_offset.astype(np.float32),  
    # float, global uv scale, default to 3000.
    'uv_scale': uv_scale,         
    
    # xyz
    'surf_cls': np.array(panel_cls, dtype=np.int32),
    'surf_mask': surf_mask.astype(bool),  # (N, H, W, 1), mask for each panel, 
                                          # N refers to number of panels in the garment. 
                                          # By default H=W=256 refers to Garmage resolution.
    'surf_wcs': surfs_wcs.astype(np.float32),   # (N, H, W, 3), panel points in world coordinate
    'surf_ncs': surfs_ncs.astype(np.float32),   # (N, H, W, 3), panel points in normalized coordinate
    
    # uv
    'surf_uv_wcs': surfs_uv_wcs.astype(np.float32),  # (N, H, W, 2)
    'surf_uv_ncs': surfs_uv_ncs.astype(np.float32),  # (N, H, W, 2)               

    # normal
    'surf_normals': surf_norms.astype(np.float32),
    'corner_normals': corner_normals.astype(np.float32),
    
#     # optional edge-related fields
#     'edge_wcs': edges_wcs.astype(np.float32),
#     'edge_ncs': edges_ncs.astype(np.float32),
#     'corner_wcs': corner_wcs.astype(np.float32),
#     'edge_uv_wcs': edges_uv_wcs.astype(np.float32),
#     'edge_uv_ncs': edges_uv_ncs.astype(np.float32),
#     'corner_uv_wcs': corner_uv_wcs.astype(np.float32),
#     'faceEdge_adj': faceEdge_adj

}
```
The `*.pkl` files are generated from raw `*.obj` garment assets and their sewing patterns saved as `panel.json` files (e.g., resources/examples/objs/0000). We use the following script to convert raw assets to `*.pkl` format:

```bash
cd data_process
python process_sxd.py -i {OBJS_SOURCE_DIR} -o {PKL_OUTPUT_DIR} --range 0,16 --use_uni_norm --nf 256
```
where `--range` indicates the input range, `--use_uni_norm` is a boolean flag for universal normalization (i.e. all the garments in the dataset will share the same global offset and scale if `--use_uni_norm` is specified) and `--nf` refers to the Garmage resolution.


## Training 
Firstly, train the VAE encoder to compress Garmages. By default, all Garmages in the dataset have a resolution of $256\times256$. Each garment is represented as a set of per-panel Garmages, forming a tensor of shape $N\times256\times256\times C$, where $C$ depends on the desired encoded fields. For instance, the simplest Garmage has four channels: the first three encode geometric positions, while the fourth (alpha) outlines the sewing pattern. For example:

```bash
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl \
    --expr stylexd_vae_surf_256_xyz_nrm_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_normals surf_mask --chunksize 512
```

Secondly, train the topology generator:

```bash
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat \
    --expr stylexd_surfpos_xyzuv_pad_repeat_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs
```

Finally, train the geometry generator:

```bash
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption
```

Training of the topology/geometry generators can run in parallel.


## Generation and Evaluation

Test the trained model with:
```bash
python src/batch_inference.py
```