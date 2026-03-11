from neural import train_epoch, optimize_model, device
from utils import save_prediction_sample, imshow
from data import validloader, trainloader
import torch


def main():
    print(
        f"Starting execution on {device}, CPU Capabilities: {torch.backends.cpu.get_cpu_capability()}"
    )
    print(optimize_model)

    train_epoch()

    if hasattr(trainloader.dataset, "dataset"):
        class_names = trainloader.dataset.dataset.classes
    else:
        class_names = trainloader.dataset.classes

    save_prediction_sample(
        dataloader=validloader,
        optimize_model=optimize_model,
        device=device,
        class_names=class_names,
        filename="examples/epoch.png",
    )

    print("Saved weights to models/aircraft_model.pth")


if __name__ == "__main__":
    main()
