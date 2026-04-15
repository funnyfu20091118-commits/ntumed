"""
PyTorch Dataset for Chest-Diffusion training.
Returns (image, report_text, labels) tuples.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset: paired (CXR image, radiology report)."""

    LABEL_COLS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
    ]

    def __init__(self, manifest_csv: str, split: str = "train", image_size: int = 256):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        # Report text
        report = str(row["report_text"])

        # CheXpert labels (for evaluation / conditioning)
        labels = []
        label_mask = []
        for col in self.LABEL_COLS:
            val = row.get(col, float("nan"))
            if pd.isna(val):
                labels.append(0.0)
                label_mask.append(0.0)
            elif val == -1.0:
                # Uncertain → ignore (mask=0); treat as negative for label value
                labels.append(0.0)
                label_mask.append(0.0)
            else:
                labels.append(1.0 if val == 1.0 else 0.0)
                label_mask.append(1.0)
        labels = torch.tensor(labels, dtype=torch.float32)
        label_mask = torch.tensor(label_mask, dtype=torch.float32)

        return img, report, labels, label_mask


class CLIPDataset(Dataset):
    """Lightweight dataset for BiomedCLIP fine-tuning (image + report pairs)."""

    def __init__(self, manifest_csv: str, split: str = "train",
                 image_preprocess=None, tokenizer=None, max_text_len: int = 256):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_preprocess = image_preprocess
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.image_preprocess is not None:
            img = self.image_preprocess(img)

        report = str(row["report_text"])
        if self.tokenizer is not None:
            tokens = self.tokenizer([report], context_length=self.max_text_len)[0]
        else:
            tokens = report

        return img, tokens
