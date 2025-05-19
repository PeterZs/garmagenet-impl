"""
# run on 187 docker
python gen_statistics_csv.py \
    --data_root_json /data/AIGP/objs_with_stitch/ \
    --data_root_obj /data/AIGP/objs_with_stitch/ \
    --output_root /data/lsr/code/style3d_gen/notebooks/data_analysize/gen_statistics_csv/output/
"""

import os
import json
import argparse
from glob import glob
from tqdm import tqdm

import trimesh
from trimesh import Trimesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_json", type=str, default="/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/notebooks/data_analysize/gen_statistics_csv/test_data_json")
    # "/data/AIGP/brep_reso_256_edge_snap_with_caption/patterns_with_caption_english/"
    parser.add_argument("--data_root_obj", type=str, default="/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/notebooks/data_analysize/gen_statistics_csv/test_data_obj")
    parser.add_argument("--output_root", type=str, default="/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/notebooks/data_analysize/gen_statistics_csv/output")
    args = parser.parse_args()

    json_list = glob(os.path.join(args.data_root_json, "**", "pattern.json"), recursive=True)
    print(f"Processing {len(json_list)} json files...")
    with (open(os.path.join(args.output_root, "styxd_data_edge.csv"), "w") as f1,
    open(os.path.join(args.output_root, "styxd_data_panel_stitch.csv"), "w") as f2):
        f1.write("edge_num\n")
        f2.write("panel_num,stitch_num\n")
        for fp in tqdm(json_list):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    AIGP_json = json.load(f)
                # edge_num = 0
                # for panel in AIGP_json["panels"]:
                #     edge_num += len(AIGP_json["panels"][0]['seqEdges'][0]["edges"])
                # f1.write(f"{edge_num}\n")
                for panel in AIGP_json["panels"]:
                    f1.write(f"{len(AIGP_json['panels'][0]['seqEdges'][0]['edges'])}\n")
                f2.write(f"{len(AIGP_json['panels'])},{len(AIGP_json['stitches'])}\n")

            except Exception as e:
                print(f"Wrone while processing json file: {fp}")
                raise NotImplementedError

    obj_list = glob(os.path.join(args.data_root_obj, "**", "*.obj"), recursive=True)
    print(f"\n\nProcessing {len(obj_list)} obj files...")
    with open(os.path.join(args.output_root, "stylexd_verts_faces.csv"), "w") as f3:
        f3.write(f"vertex_num,face_num\n")
        for fp in tqdm(obj_list):
            try:
                mesh = trimesh.load(fp, force="mesh", process=False)
                f3.write(f"{len(mesh.vertices)},{len(mesh.faces)}\n")
            except Exception as e:
                print(f"Wrone while processing obj file: {fp}")
                raise NotImplementedError
    # # Number of edges per panel
    # stylexd_data = pd.read_csv("styxd_data_edge.csv")["edge_num"]
    # garmentcode_data = pd.read_csv("garmentcode_data_edge.csv")["edge_num"]
    # draw_hist_comp("edges/panel", stylexd_data, garmentcode_data, x_lim=50)
    #
    # # Number of panels per pattern
    # stylexd_data = pd.read_csv("styxd_data_panel_stitch.csv")["panel_num"]
    # garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["panel_num"]
    # draw_hist_comp("panels/pattern", stylexd_data, garmentcode_data, x_lim=30)
    #
    # # Number of stitches per pattern
    # stylexd_data = pd.read_csv("styxd_data_panel_stitch.csv")["stitch_num"]
    # garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["stitch_num"]
    # draw_hist_comp("stitches/panel", stylexd_data, garmentcode_data, x_lim=125)
    #
    # # Number of vertices per garment
    # stylexd_data = pd.read_csv("stylexd_verts_faces.csv")["vertex_num"]
    # garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["vertex_num"]
    # draw_hist_comp("vertices/garment", stylexd_data, garmentcode_data, x_lim=100000)
    #
    # # Number of faces per garment
    # stylexd_data = pd.read_csv("stylexd_verts_faces.csv")["face_num"]
    # garmentcode_data = pd.read_csv("garmentcode_data_panel_stitch.csv")["face_num"]
    # draw_hist_comp("faces/garment", stylexd_data, garmentcode_data, x_lim=60000)
