#%%
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have these from your test set evaluation:
# - `y_true`: true class labels (1D list or tensor)
# - `y_pred`: predicted class labels (1D list or tensor)
# - `class_names`: list of class labels in order (e.g., ['airplane', 'automobile', ..., 'truck'])
#%%
data = torch.load("eval_probs_targets.pth") #probs: Softmax confidence scores
probs = data["probs"]
labels = data["targets"]
confidences, preds = torch.max(probs, dim=1)    # Top confidence + predicted label

y_true = labels
y_pred = preds
#%%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
#%%
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted')

# Annotate 
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_mtx_img.png")
plt.show()

#%% 
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_mtx_heatmap.png")
plt.show()

# %%
