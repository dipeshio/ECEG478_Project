import torch, random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL   import Image
from pathlib import Path

# ── transforms ──────────────────────────────────────────────
IMG_SIZE = 224
train_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ── dataset ─────────────────────────────────────────────────
class FairFaceMulti(Dataset):
    def __init__(self, df, img_dir, train=True):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = Path(img_dir)
        self.tfm      = train_tfms if train else val_tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img_p = self.img_dir / Path(row.file).name   # robust join
        img   = Image.open(img_p).convert("RGB")
        age    = torch.tensor(row.age_label,    dtype=torch.long)
        gender = torch.tensor(row.gender_label, dtype=torch.long)
        race = torch.tensor(row.race_label, dtype=torch.long)

        return self.tfm(img), {"age": age, "gender": gender, "race": race}

    def __str__(self):
        return f"FairFaceMulti(n={len(self)}, dir={self.img_dir})"

