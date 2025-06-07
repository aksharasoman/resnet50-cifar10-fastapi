import torch
import torchvision
import matplotlib.pyplot as plt

# CIFAR-10 classes
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def show_misclassified(
    all_images, all_labels, all_preds, all_probs,
    true_label, pred_label, max_samples=6
):
    # Select matching misclassified samples
    indices = torch.where((all_labels == true_label) & (all_preds == pred_label))[0]
    if len(indices) == 0:
        print("No matching misclassified samples found.")
        return

    # Sort by confidence (highest first)
    selected_probs = all_probs[indices, pred_label]
    sorted_idx = torch.argsort(selected_probs, descending=True)
    indices = indices[sorted_idx][:max_samples]

    # Plot
    num_images = len(indices)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 1.5, 3)) # Using 2 inches per image horizontally
    for i, idx in enumerate(indices):
        img = all_images[idx]
        label = all_labels[idx].item()
        pred = all_preds[idx].item()
        conf = all_probs[idx, pred].item()

        axs[i].imshow(img)
        axs[i].set_title(f"GT: {cifar_classes[label]}\nPred: {cifar_classes[pred]}\nConf: {conf:.2f}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"misclassified_{cifar_classes[label]}_Pred:{cifar_classes[pred]}.png")

def get_all_images_pil():
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None  # Raw PIL Images
    )
    return [img for img, _ in dataset]  # list of PIL.Image.Image

if __name__ == "__main__":
    # Load saved outputs
    data = torch.load("eval_probs_targets.pth")
    probs = data["probs"]
    labels = data["targets"]
    confidences, preds = torch.max(probs, dim=1)

    # Get all images of test set:
    all_images = get_all_images_pil()

    # Now visualize confusing pairs
    show_misclassified(all_images, labels, preds, probs, true_label=3, pred_label=5)  # cat → dog
    show_misclassified(all_images, labels, preds, probs, true_label=5, pred_label=3)  # dog → cat
