from neural import train_epoch, optimize_model, device
from utils import save_prediction_sample
from data import validloader, trainloader
import torch


def main():
    print(
        f"Starting execution on {device}, CPU Capabilities: {torch.backends.cpu.get_cpu_capability()}"
    )
    print(optimize_model)

    train_epoch()

    save_prediction_sample(validloader, optimize_model, device, f"examples/epoch.png")

    print("Saved weights to models/aircraft_model.pth")


if __name__ == "__main__":
    main()
