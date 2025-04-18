import os
import pickle
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from src.vis import draw_bbox_geometry, draw_bbox_geometry_3D2D
from src.vis import _create_bounding_box_lines, _create_bounding_box_mesh
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def vis(data, data_path):
    n_surfs = len(data["surf_bbox_wcs"])
    colormap = plt.cm.coolwarm
    colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]

    surf_uv_bbox_wcs = np.zeros((n_surfs, 6))
    surf_uv_bbox_wcs[:,[0,1,3,4]] = data["surf_uv_bbox_wcs"]
    points_uv = np.concatenate([data["surf_uv_wcs"],
                    np.zeros((data["surf_uv_wcs"].shape[0],
                              data["surf_uv_wcs"].shape[1],
                              data["surf_uv_wcs"].shape[2], 1))], axis=-1)

    fig = draw_bbox_geometry_3D2D(
        bboxes=[data["surf_bbox_wcs"],surf_uv_bbox_wcs],
        bbox_colors=colors,
        points=[data["surf_wcs"], points_uv],
        point_masks=data["surf_mask"],
        point_colors=colors,
        num_point_samples=1000,
        title=f"{os.path.basename(data_path)}: {data['caption']}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )
    return fig
    # draw_bbox_geometry(
    #     bboxes=,
    #     bbox_colors=colors,
    #     title=f"{os.path.basename(data_path)}: {data['caption']}",
    #     points=,
    #     point_masks=data["surf_mask"],
    #     point_colors=colors,
    #     num_point_samples=1000,
    #     # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
    #     show_num=True,
    #     fig_show="browser"
    # )

def load_data(data_path):
    with open(data_path, 'rb') as f:
        d = pickle.load(f)
    data = {
        "surf_wcs": d["surf_wcs"],
        "surf_uv_wcs": d["surf_uv_wcs"],
        "surf_ncs": d["surf_ncs"],
        "surf_uv_ncs": d["surf_uv_ncs"],

        "surf_bbox_wcs": d["surf_bbox_wcs"],
        "surf_uv_bbox_wcs": d["surf_uv_bbox_wcs"],
        "surf_mask": d["surf_mask"],
        "caption": d["caption"],
    }
    data["bbox_mask"] = np.ones((len(d["surf_cls"])), dtype=np.bool)
    return data

def bbox_segmentation(data):
    # BBox分割的逻辑应该是
    # 1.轴，2.百分比
    a=1
    n_surf = len(data['surf_uv_wcs'])
    panel_id, seg_orient, seg_param = map(str, input("Panel_id Axis Param\n").split())
    # panel_id = 5
    panel_id = min(max(int(panel_id), 0), n_surf-1)
    seg_param = float(seg_param)
    seg_param = min(max(seg_param, 0), 1)
    if seg_orient=="x":
        axis=0
    elif seg_orient=="y":
        axis=1
    else:
        return data

    bbox_3D = data["surf_bbox_wcs"]
    data["surf_bbox_wcs"] = np.delete(data["surf_bbox_wcs"], panel_id, axis=0)
    new_bbox_3D_0 = deepcopy(bbox_3D[panel_id])
    new_bbox_3D_1 = deepcopy(bbox_3D[panel_id])
    new_bbox_3D_0[axis+3] = seg_param * new_bbox_3D_0[axis] + (1-seg_param)*new_bbox_3D_0[axis+3]
    new_bbox_3D_1[axis] = seg_param * new_bbox_3D_1[axis] + (1-seg_param)*new_bbox_3D_1[axis+3]
    data["surf_bbox_wcs"] = np.insert(data["surf_bbox_wcs"], panel_id, new_bbox_3D_0, axis=0)
    data["surf_bbox_wcs"] = np.insert(data["surf_bbox_wcs"], panel_id, new_bbox_3D_1, axis=0)

    bbox_2D = data["surf_uv_bbox_wcs"]
    data["surf_uv_bbox_wcs"] = np.delete(data["surf_uv_bbox_wcs"], panel_id, axis=0)
    new_bbox_2D_0 = deepcopy(bbox_2D[panel_id])
    new_bbox_2D_1 = deepcopy(bbox_2D[panel_id])
    new_bbox_2D_0[axis+2] = seg_param * new_bbox_2D_0[axis] + (1-seg_param)*new_bbox_2D_0[axis+2]
    new_bbox_2D_1[axis] = seg_param * new_bbox_2D_1[axis] + (1-seg_param)*new_bbox_2D_1[axis+2]
    data["surf_uv_bbox_wcs"] = np.insert(data["surf_uv_bbox_wcs"], panel_id, new_bbox_2D_0, axis=0)
    data["surf_uv_bbox_wcs"] = np.insert(data["surf_uv_bbox_wcs"], panel_id, new_bbox_2D_1, axis=0)

    data["surf_mask"][panel_id] = False
    data["surf_mask"] = np.insert(data["surf_mask"], panel_id, np.zeros_like(data["surf_mask"][panel_id], dtype=np.bool), axis=0)

    data["surf_wcs"][panel_id] = 0
    data["surf_wcs"] = np.insert(data["surf_wcs"], panel_id, np.zeros_like(data["surf_wcs"][panel_id]), axis=0)
    data["surf_uv_wcs"][panel_id] = 0
    data["surf_uv_wcs"] = np.insert(data["surf_uv_wcs"], panel_id, np.zeros_like(data["surf_uv_wcs"][panel_id]), axis=0)

    data["surf_ncs"][panel_id] = 0
    data["surf_ncs"] = np.insert(data["surf_ncs"], panel_id, np.zeros_like(data["surf_ncs"][panel_id]), axis=0)
    data["surf_uv_ncs"][panel_id] = 0
    data["surf_uv_ncs"] = np.insert(data["surf_uv_ncs"], panel_id, np.zeros_like(data["surf_uv_ncs"][panel_id]), axis=0)

    data["bbox_mask"][panel_id] = False
    data["bbox_mask"] = np.insert(data["bbox_mask"], panel_id, False, axis=0)
    return data

def save_results(out_dir, data, fig):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump(data, f)
    if fig is not None:
        fig.write_html(os.path.join(out_dir, "vis.html"))


if __name__ == '__main__':
    data_root = "data/256/brep_reso_256_edge_snap_with_caption"
    output_dir = "_LSR/gen_experiment_data/bbox_editing/output"

    def sort_key(x): return os.path.basename(x).split(".")[0]
    data_list = sorted(glob(os.path.join(data_root, "*.pkl")), key=sort_key)

    fig = None
    history = []

    # 是否从某个数据开始
    start_from = "00000.pkl"
    if start_from is not None: start_index = [os.path.basename(d) for d in data_list].index(start_from)
    else: start_index = 0
    for data_idx in range(start_index, len(data_list)):
        out_dir = os.path.join(output_dir,f"{data_idx}".zfill(5))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(data_list[data_idx], out_dir)

        data_path = data_list[data_idx]
        orig_data = load_data(data_path)

        data = deepcopy(orig_data)
        fig_orig = vis(data, data_path)
        fig_orig.write_html(os.path.join(out_dir, "vis.html"))
        history.append(data)

        while True:
            print("\n")
            # 读取一个Garmage
            user_input = input("Operates:\n"
                               "1.Seg\n"
                               "#r recovery\n"
                               "#s save\n"
                               "#q next\n")
            if "1" in user_input or "seg" in user_input.lower():
                print("Segmentation")
                history.append(data)
                data = deepcopy(data)
                data = bbox_segmentation(data)
                fig = vis(data, data_path)
                continue
            elif "#r" in user_input.lower():
                if len(history)>0:
                    data = history[-1]
                    del history[-1]
                fig = vis(data, data_path)
                continue
            elif "#s" in user_input.lower():
                print("Save")
                datatype = input("datatype: 1 seg\n")
                if int(datatype) == 1:
                    datatype = "bbox_seg"
                else:
                    continue
                save_results(os.path.join(out_dir, datatype), data, fig)
                continue
            elif "#q" in user_input.lower():
                break