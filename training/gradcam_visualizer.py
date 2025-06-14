'''
Here’s a complete and reusable Grad-CAM script that:
Loads your saved ResNet-50 model and predictions.
Identifies misclassified samples.
Generates Grad-CAM visualizations for both:Predicted class & True class
Saves them side-by-side in a single image.
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import os

# Constants
MODEL_PATH = "dev_stage_code/best_model.pth"
EVAL_DATA_PATH = "dev_stage_code/best_eval_probs_targets.pth"
SAVE_DIR = "cam_outputs"
NUM_SAMPLES = 10
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Transforms
normalize = T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
inv_normalize = T.Normalize(mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
                            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010])
transform = T.Compose([T.ToTensor(), normalize])

# Load CIFAR-10 test set
test_set = CIFAR10(root="./data", train=False, transform=transform, download=True)

# Load model
model = models.resnet50() #no weight specified => no pretrained model loaded
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 10)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Grad-CAM setup
cam_extractor = GradCAM(model, target_layer="layer4")

# Load eval outputs
data = torch.load(EVAL_DATA_PATH)
probs = data["probs"]
labels = data["targets"]
confidences, preds = torch.max(probs, dim=1)

# Get misclassified indices
misclassified_idxs = (preds != labels).nonzero(as_tuple=True)[0]


def generate_cam_image(idx):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Get image and ground truth
    img_tensor, true_label = test_set[idx]
    input_tensor = img_tensor.unsqueeze(0)
    pred = model(input_tensor).argmax(dim=1).item()

    # Unnormalize image
    img_vis = inv_normalize(img_tensor)
    img_pil = to_pil_image(img_vis)

    # Grad-CAM for predicted class
    cam_pred = cam_extractor(pred, model(input_tensor))[0]
    cam_pred = to_pil_image(cam_pred.squeeze(0), mode='F')
    overlay_pred = overlay_mask(img_pil, cam_pred, alpha=0.5)

    # Grad-CAM for true class
    cam_true = cam_extractor(true_label, model(input_tensor))[0]
    cam_true = to_pil_image(cam_true.squeeze(0), mode='F')
    overlay_true = overlay_mask(img_pil, cam_true, alpha=0.5)

    # Annotate and combine
    def annotate(image, text):
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.text((5, 5), text, fill="white")
        return annotated

    annotated_pred = annotate(overlay_pred, f"Pred: {CLASSES[pred]}")
    annotated_true = annotate(overlay_true, f"True: {CLASSES[true_label]}")

    combined = Image.new("RGB", (annotated_pred.width * 2, annotated_pred.height))
    combined.paste(annotated_pred, (0, 0))
    combined.paste(annotated_true, (annotated_pred.width, 0))

    # Save image
    combined.save(os.path.join(SAVE_DIR, f"idx{idx}_true{CLASSES[true_label]}_pred{CLASSES[pred]}.png"))


if __name__ == "__main__":
    print(f"Generating Grad-CAMs for {NUM_SAMPLES} misclassified samples...")
    for idx in misclassified_idxs[:NUM_SAMPLES]:
        generate_cam_image(idx.item())
    print(f"Saved visualizations to '{SAVE_DIR}/'")
