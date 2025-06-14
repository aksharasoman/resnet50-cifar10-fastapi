'''
A complete, reusable Grad-CAM script using the pytorch-grad-cam library:
1. Loads your best_model.pth ResNet-50 with partial fine-tuning.
2. Loads your best_eval_probs_targets.pth for predictions.
3. Identifies misclassified samples.
4. Uses Grad-CAM to visualize both predicted and true classes.
5. Saves side-by-side CAMs with the original image.
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import preprocess_image

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CONFIG ----------
MODEL_PATH = "dev_stage_code/best_model.pth"
EVAL_PATH = "dev_stage_code/best_eval_probs_targets.pth"
SAVE_DIR = "gradcam_outputs"
NUM_SAMPLES = 20  # number of misclassified samples to visualize
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD MODEL ----------
def load_model():
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 10)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

# ---------- LOAD DATA ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
unnormalize = transforms.Normalize(mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                                   std=[1/0.2023, 1/0.1994, 1/0.2010])

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# ---------- LOAD PREDICTIONS ----------
data = torch.load(EVAL_PATH, map_location=device)
probs = data["probs"]
labels = data["targets"]
confidences, preds = torch.max(probs, dim=1)
misclassified_idxs = torch.where(preds != labels)[0][:NUM_SAMPLES]

# ---------- PREPARE CAM ----------
def get_cam(model, target_layer, input_tensor, target_category):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])
    return grayscale_cam[0]

# ---------- VISUALIZE ----------
def visualize_sample(model, idx, save_path):
    img_tensor, label = test_dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    pred = preds[idx].item()
    true = labels[idx].item()

    target_layer = model.layer4[-1]

    # Grad-CAM
    cam_pred = get_cam(model, target_layer, input_tensor, target_category=pred)
    cam_true = get_cam(model, target_layer, input_tensor, target_category=true)

    # Unnormalize and resize image for better visualization
    img = unnormalize(img_tensor).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img_resized = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    # Resize CAMs
    cam_pred_resized = np.array(Image.fromarray((cam_pred * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    cam_true_resized = np.array(Image.fromarray((cam_true * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    cam_image_pred = show_cam_on_image(img_resized, cam_pred_resized, use_rgb=True)
    cam_image_true = show_cam_on_image(img_resized, cam_true_resized, use_rgb=True)

    # Combine horizontally
    combined = np.hstack((cam_image_pred, cam_image_true))
    combined_image = Image.fromarray(combined)

    # Add text
    draw = ImageDraw.Draw(combined_image)
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((10, 5), f"Predicted: {pred}", fill="white", font=font)
    draw.text((240, 5), f"True: {true}", fill="white", font=font)

    combined_image.save(save_path)

# ---------- MAIN ----------
def main():
    model = load_model()
    print(f"Generating Grad-CAM for {len(misclassified_idxs)} misclassified samples...")
    for i, idx in enumerate(misclassified_idxs):
        save_path = os.path.join(SAVE_DIR, f"cam_{i:03d}_true{labels[idx]}_pred{preds[idx]}.png")
        visualize_sample(model, idx.item(), save_path)
    print("Done.")

if __name__ == "__main__":
    main()
