import torch
import matplotlib.pyplot as plt

def per_class_accuracy(y_true, y_pred, class_names):
    n_classes = len(class_names)
    acc_per_class = {}
    for i in range(n_classes):
        correct = ((y_true == i) & (y_pred == i)).sum()
        total = (y_true == i).sum()
        acc_per_class[class_names[i]] = 100.0 * correct / total if total > 0 else 0.0
    return acc_per_class

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

data = torch.load("eval_probs_targets.pth") #probs: Softmax confidence scores
probs = data["probs"]
val_labels = data["targets"]
confidences, val_preds = torch.max(probs, dim=1)    # Top confidence + predicted label

# train_accs = per_class_accuracy(train_labels, train_preds, class_names)
val_accs = per_class_accuracy(val_labels, val_preds, class_names)

# Compare them
print("| CLASS | Val. Acc |")
print("|---|---|")
for cls in class_names:
    # print(f"{cls:10} | Train Acc: {train_accs[cls]:.2f}% | Val Acc: {val_accs[cls]:.2f}%")
    print(f"|{cls:10} | {val_accs[cls]:.2f}%|")


# Prepare data for plotting
acc_values = [val_accs[cls] for cls in class_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, acc_values, color='skyblue')

# Annotate accuracy on top of each bar
for bar, acc in zip(bars, acc_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.2f}%", 
                ha='center', va='bottom', fontsize=9)

plt.ylabel('Validation Accuracy (%)')
plt.xlabel('Class')
plt.title('Per-Class Validation Accuracy')
plt.ylim(0, 110)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("accuracy_per_class.png")


