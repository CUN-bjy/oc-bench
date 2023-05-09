# modified from https://github.com/singhgautam/steve

import os
import glob
import torch

import tensorflow_datasets as tfds
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ocbench.dataset import DATASET_PATH
from tqdm import tqdm


def downloadMoviDataset(path_to_download, level, size, phase):
    ds, ds_info = tfds.load(
        f"movi_{level}/{size}x{size}:1.0.0",
        data_dir="gs://kubric-public/tfds",
        with_info=True,
    )
    train_iter = iter(tfds.as_numpy(ds[phase]))

    to_tensor = transforms.ToTensor()

    b = 0
    for record in train_iter:
        video = record["video"]
        T, *_ = video.shape

        # setup dirs
        path_vid = os.path.join(path_to_download, f"{b:08}")
        os.makedirs(path_vid, exist_ok=True)

        for t in tqdm(range(T)):
            img = video[t]
            img = to_tensor(img)
            vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))

        b += 1


class GlobVideoDataset(Dataset):
    def __init__(self, level, phase, img_size, ep_len=3, img_glob="*.png"):
        self.root = os.path.join(
            DATASET_PATH, f"movi_{level}", f"{img_size}x{img_size}", f"{phase}"
        )
        if not os.path.exists(self.root):
            downloadMoviDataset(self.root, level, img_size, phase)
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(self.root + "/*"))
        self.ep_len = ep_len

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--level", default="A", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--ep_len", type=int, default=3)

    args = parser.parse_args()

    train_dataset = GlobVideoDataset(
        level=args.level,
        phase="train",
        img_size=args.image_size,
        ep_len=args.ep_len,
        img_glob="????????_image.png",
    )
    val_dataset = GlobVideoDataset(
        level=args.level,
        phase="validation",
        img_size=args.image_size,
        ep_len=args.ep_len,
        img_glob="????????_image.png",
    )
