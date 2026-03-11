import torch
from data import dataloader
from neural import NeuralNetwork, device

model = NeuralNetwork().to(device)

for X, y in dataloader:
    X, y = X.to(device), y.to(device)

    pred = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Predictions shape: {pred.shape}")
    break
