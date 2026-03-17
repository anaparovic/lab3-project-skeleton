import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.custom_net import CustomNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

project_root = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(project_root, "dataset", "tiny-imagenet-200", "val")
checkpoint_path = os.path.join(project_root, "checkpoints", "best_model.pth")

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = ImageFolder(val_dir, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

model = CustomNet(num_classes=200).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

correct = 0
total = 0

all_images = []
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_images.extend(images.cpu())
        all_labels.extend(labels.cpu())
        all_preds.extend(predicted.cpu())

acc = 100 * correct / total
print(f"Validation Accuracy: {acc:.2f}%")

classes = val_dataset.classes

plt.figure(figsize=(12, 12))
indices = random.sample(range(len(all_images)), 9)

for i, idx in enumerate(indices):
    img = all_images[idx].permute(1, 2, 0).numpy()
    true_label = classes[all_labels[idx]]
    pred_label = classes[all_preds[idx]]

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(project_root, "checkpoints", "eval_examples.png"))
plt.show()