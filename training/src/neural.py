import os

# Threading optimization for 6 physical cores on the 5600X
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import numpy as np
import torch
from torch import nn

torch.set_num_threads(6)
from data import (
    validloader,
    trainloader,
    batch_size,
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


model = NeuralNetwork().to(device).to(memory_format=torch.channels_last)

optimize_model = torch.compile(model, mode="default", dynamic=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(optimize_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

scaler_type = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=scaler_type)

epochs = 10
min_valid_loss = np.inf


def train_epoch():
    global min_valid_loss

    print("Warming compiler...")

    dummy_data = torch.randn(32, 3, 32, 32).to(
        device, memory_format=torch.channels_last
    )
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        _ = optimize_model(dummy_data)

    for e in range(epochs):
        train_loss = 0.0
        optimize_model.train()

        train_bar = tqdm(
            trainloader, desc=f"Epoch {e+1}/{epochs} [Train]", unit="batch"
        )

        for data, labels in train_bar:
            data = data.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                target = optimize_model(data)
                loss = criterion(target, labels)

            if scaler_type:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        valid_loss = 0.0
        optimize_model.eval()

        valid_bar = tqdm(
            validloader, desc=f"Epoch {e+1}/{epochs} [Valid]", unit="batch", leave=False
        )

        with torch.no_grad():
            for data, labels in valid_bar:
                data = data.to(device, memory_format=torch.channels_last)
                labels = labels.to(device)
                target = optimize_model(data)
                loss = criterion(target, labels)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(trainloader)
        avg_valid_loss = valid_loss / len(validloader)

        print(
            f"\nEpoch {e+1} summary: Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}"
        )

        if min_valid_loss > avg_valid_loss:
            print(
                f"Validation Loss Decreased ({min_valid_loss:.6f} ---> {avg_valid_loss:.6f}) - Saving Model"
            )
            min_valid_loss = avg_valid_loss
            torch.save(optimize_model.state_dict(), "models/aircraft_model.pth")


if __name__ == "__main__":
    train_epoch()
