import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class LOLDataset(Dataset):
    """
    Dataset loader for LOL (Low-Light) dataset.
    Pairs low-light and normal-light (high) images.
    """
    def __init__(self, low_dir, high_dir, patch_size=256, is_train=True):
        super().__init__()
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.patch_size = patch_size
        self.is_train = is_train

        # Get list of low images (.png)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.filenames = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(valid_exts) and not f.startswith('.')
        ])

        # Verify corresponding high images exist
        high_files_set = set(os.listdir(high_dir))
        self.valid_pairs = []
        for f in self.filenames:
            if f in high_files_set:
                self.valid_pairs.append(f)
            else:
                print(f"[Warning] Missing matching ground truth for {f} in {high_dir}")

        print(f"Loaded {len(self.valid_pairs)} pairs from low: {low_dir} and high: {high_dir} (is_train={is_train})")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        filename = self.valid_pairs[idx]
        low_path = os.path.join(self.low_dir, filename)
        high_path = os.path.join(self.high_dir, filename)

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        if self.is_train and self.patch_size is not None:
            w, h = low_img.size
            th, tw = self.patch_size, self.patch_size
            if w > tw and h > th:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
                low_img = TF.crop(low_img, i, j, th, tw)
                high_img = TF.crop(high_img, i, j, th, tw)
            elif w != tw or h != th:
                low_img = TF.resize(low_img, (th, tw))
                high_img = TF.resize(high_img, (th, tw))

            # Random horizontal flipping
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

            # Random vertical flipping
            if random.random() > 0.5:
                low_img = TF.vflip(low_img)
                high_img = TF.vflip(high_img)

        # Convert to tensor [0, 1]
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        return low_tensor, high_tensor, filename
