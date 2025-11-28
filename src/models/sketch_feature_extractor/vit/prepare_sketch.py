"""
prepare sketch feature laion2b
"""

import argparse
from tqdm import tqdm
from pathlib import Path

import timm
import torch
import numpy as np
from PIL import Image
from safetensors import safe_open
from torch.utils.data import DataLoader

from src.utils import ensure_directory
from utils.sketch_utils import _transform


VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'


class FeatureExtractorDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: str, image_resolution: int = 224):
        super().__init__()
        self.image_folder = Path(image_folder)
        # self.image_paths = [p for p in self.image_folder.rglob('*.png')]
        self.image_paths = [p for p in self.image_folder.rglob('*.png') if "_0.png" in str(p)]
        self.preprocess = _transform(image_resolution)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image), str(image_path)


def extract_features(dataset_folder: str, feature_output_folder: str, safetensors_path: str, image_resolution: int = 224):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = FeatureExtractorDataset(dataset_folder, image_resolution)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    model = timm.create_model(VIT_MODEL, pretrained=False).to(device)
    model.eval()

    with safe_open(safetensors_path, framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict)

    ensure_directory(feature_output_folder)

    print("Processing sketches...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.to(device)
            features = model.forward_features(images).squeeze().cpu().numpy()
            for feature, path in zip(features, paths):
                relative_path = Path(path).relative_to(dataset_folder)
                feature_path = Path(feature_output_folder) / relative_path.with_suffix('.npy')
                ensure_directory(feature_path.parent)

                feature_save = feature[0]
                np.save(feature_path, feature_save)
                print(f"Saved feature to {feature_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str,
                        default="src/models/sketch_feature_extractor/vit/test_data")
    parser.add_argument('--feature_output_folder', type=str,
                        default="src/models/sketch_feature_extractor/vit/output")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    feature_output_folder = args.feature_output_folder
    safetensors_path = 'path-to-<models--timm--vit_huge_patch14_clip_224.laion2b>'
    extract_features(dataset_folder, feature_output_folder, safetensors_path)
