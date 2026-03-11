import matplotlib.pyplot as plt
import torch
import numpy as np


def check_sparsity(model):
    """Calculates the percentage of near-zero weights in the classifier's first layer."""
    layer = model._orig_mod.classifier[0]
    weights = layer.weight.data

    zero_weights = torch.sum(torch.abs(weights) < 1e-5).item()
    total_weights = weights.numel()
    sparsity = (zero_weights / total_weights) * 100

    print(
        f"Layer 1 Sparsity: {sparsity:.2f}% ({zero_weights}/{total_weights} dead weights)"
    )
    return sparsity


def imshow(img_tensor):
    """Un-normalizes and prepares a tensor for matplotlib plotting."""
    img = img_tensor.clone().detach().cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    return img.permute(1, 2, 0).numpy()


def save_prediction_sample(
    dataloader, optimize_model, device, class_names, filename="prediction_sample.png"
):
    optimize_model.eval()

    images, labels = next(iter(dataloader))
    images_gpu = images.to(device, memory_format=torch.channels_last)

    with torch.no_grad():
        output = optimize_model(images_gpu)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidences, preds = torch.max(probs, 1)

    preds, labels, confs = preds.cpu(), labels.cpu(), confidences.cpu()

    fig, axes = plt.subplots(4, 4, figsize=(15, 16))
    axes = axes.flatten()

    for i in range(min(16, len(images))):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(imshow(images[i]))

        p_idx, a_idx = preds[i].item(), labels[i].item()
        conf = confs[i].item() * 100
        color = "green" if p_idx == a_idx else "red"

        label_text = (
            f"P: {class_names[p_idx][:22]} ({conf:.1f}%)\nA: {class_names[a_idx][:22]}"
        )

        ax.text(
            0.5,
            -0.15,
            label_text,
            color=color,
            fontsize=7,
            ha="center",
            va="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
        )

    plt.subplots_adjust(hspace=0.4, wspace=0.1, bottom=0.05)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Sample predictions saved to {filename}")
    plt.close()
