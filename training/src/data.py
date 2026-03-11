from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128

# 1. Define transforms FIRST
transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = datasets.FGVCAircraft(
    root="training/data", download=True, transform=transforms
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
