import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

num_workers = os.cpu_count()
batch_size = 64
img_size = 64

pin_memory = torch.cuda.is_available()


def remove_aircraft_banner(img):
    """Removes the 20px copyright banner from the bottom of FGVC images."""
    return img.crop((0, 0, img.width, img.height - 20))


data_transforms = transforms.Compose(
    [
        transforms.Lambda(remove_aircraft_banner),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

train_set = datasets.FGVCAircraft(
    root="training/data", split="train", download=True, transform=data_transforms
)

valid_set = datasets.FGVCAircraft(
    root="training/data", split="val", download=True, transform=data_transforms
)

trainloader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=pin_memory,
    persistent_workers=True if num_workers > 0 else False,
)

validloader = DataLoader(
    dataset=valid_set,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin_memory,
    persistent_workers=True if num_workers > 0 else False,
)
