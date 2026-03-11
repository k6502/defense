import matplotlib.pyplot as plt
import torch


def save_prediction_sample(
    dataloader, optimize_model, device, filename="prediction_sample.png"
):
    optimize_model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        output = optimize_model(images)
        _, preds = torch.max(output, 1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    fig = plt.figure(figsize=(12, 12))
    for i in range(min(16, len(images))):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)

        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"P: {preds[i].item()} | A: {labels[i].item()}", color=color)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Sample predictions saved to {filename}")
    plt.close()
