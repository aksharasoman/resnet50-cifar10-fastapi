# use it: reliability_diagram(probs, labels, save_path="reliability_diagram.png")
import torch
import matplotlib.pyplot as plt

def reliability_diagram(probs, labels, n_bins=15, save_path=None):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0

    avg_confs = []
    avg_accs = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            avg_confs.append(conf_in_bin.item())
            avg_accs.append(acc_in_bin.item())
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(avg_confs, avg_accs, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE: {ece.item():.4f})')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    data = torch.load("eval_probs_targets.pth") #probs: Softmax confidence scores
    probs = data["probs"]
    labels = data["targets"]
    reliability_diagram(probs, labels, save_path="reliability_diagram.png")
