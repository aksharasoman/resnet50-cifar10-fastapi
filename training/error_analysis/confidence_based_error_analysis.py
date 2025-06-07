'''
1. Plot confidence distribution of correct vs incorrect predictions
2. Plot confidence per class
'''
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
data = torch.load("eval_probs_targets.pth") #probs: Softmax confidence scores
probs = data["probs"]
labels = data["targets"]
confidences, preds = torch.max(probs, dim=1)    # Top confidence + predicted label
correct = preds.eq(labels)               # Boolean tensor

# Separate confidence values
confidences_correct = confidences[correct].tolist()
confidences_incorrect = confidences[~correct].tolist()

# Count overconfident wrong predictions : Incorrect predictions with confidence > 0.9
high_conf_wrong = (confidences[~correct] > 0.9).sum().item()
total_wrong = (~correct).sum().item()
#%%

class_confidences = [[] for _ in range(10)]  # CIFAR-10 has 10 classes 
class_counts = [0] * 10
#%%

# Collect per-class confidences
for i in range(len(labels)):
    class_confidences[labels[i].item()].append(confidences[i].item()) #list of lists
    class_counts[labels[i].item()] += 1

# Collect per-class confidences for correct and incorrect predictions separately
class_confidences_correct = [[] for _ in range(10)]
class_confidences_incorrect = [[] for _ in range(10)]

for i in range(len(labels)):
    label = labels[i].item()
    conf = confidences[i].item()
    if correct[i]:
        class_confidences_correct[label].append(conf)
    else:
        class_confidences_incorrect[label].append(conf)
# Calculate number of incorrect classifications per class
num_incorrect_per_class = [len(c) for c in class_confidences_incorrect]

#%%
# --- Stats ---
print(f"\nAverage confidence (correct):   {np.mean(confidences_correct):.4f}")
print(f"Average confidence (incorrect): {np.mean(confidences_incorrect):.4f}")
if total_wrong > 0:
    print(f"High-confidence wrong predictions (>0.9): {high_conf_wrong} / {total_wrong} "
            f"({100.0 * high_conf_wrong / total_wrong:.2f}%)")

#%%
# Plot histograms
plt.figure(figsize=(8, 5))
plt.hist(confidences_correct, bins=20, alpha=0.6, label='Correct', color='green')
plt.hist(confidences_incorrect, bins=20, alpha=0.6, label='Incorrect', color='red')
plt.xlabel('Model Confidence')
plt.ylabel('Number of Samples')
plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("confidence_distribution.png")

#%%
#---- Standalone Plot of Incorrect predictions' confidence distribution -----#
plt.figure(figsize=(8, 5))
plt.hist(confidences_incorrect,bins=20,alpha=0.6,label="Incorrect",color="red")
plt.xlabel('Model Confidence')
plt.ylabel('Number of Samples')
plt.title('Confidence Distribution: Incorrect Predictions')
plt.grid(True)
plt.tight_layout()
plt.savefig("incorrect_preds_confidence_distribution.png")


#%%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

avg_conf_per_class = [np.mean(c) if c else 0 for c in class_confidences]
avg_conf_incorrect_per_class = [np.mean(c) if c else 0 for c in class_confidences_incorrect]
classes = class_names if class_names else list(range(10))

plt.figure(figsize=(10, 5))
plt.bar(classes, avg_conf_per_class, color='skyblue')
plt.ylabel('Average Confidence')
plt.title('Average Model Confidence per Class')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('confidence_per_class.png')
plt.show()

# Plot average confidence for incorrect predictions per class
plt.figure(figsize=(10, 5))
plt.bar(classes, avg_conf_incorrect_per_class, color='salmon')
plt.ylabel('Average Confidence (Incorrect)')
plt.title('Average Model Confidence for Incorrect Predictions per Class')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('incorrect_confidence_per_class.png')
plt.show()


# %%
# Plot number of incorrect classifications per class
plt.figure(figsize=(10, 5))
bars = plt.bar(classes, num_incorrect_per_class, color='orange')
plt.ylabel('Number of Incorrect Predictions')
plt.title('Number of Incorrect Classifications per Class')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
# Display the number at the top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontsize=10)
plt.savefig('num_incorrect_per_class.png')
plt.show()

