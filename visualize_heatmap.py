import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from resnet import resnet18
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = torch.device('cpu')


# Load the trained model
model = resnet18()
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1)  # Binary classification, so 1 output unit
)
model.to(device)
model.load_state_dict(torch.load('deepfake_classifier_ben.pth', map_location=torch.device('cpu')))
model.eval()

def grad_cam(model, image_path, target_layer):
    """
    Generate a Grad-CAM heatmap for a specific image and model layer,
    and return the model's prediction.
    """
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    model.eval()
    input_tensor.requires_grad_(True)

    def get_features_hook(module, input, output):
        global features
        features = output.detach()

    def get_gradients_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0].detach()

    target_layer_module = model.get_submodule(target_layer)
    feature_handle = target_layer_module.register_forward_hook(get_features_hook)
    gradient_handle = target_layer_module.register_full_backward_hook(get_gradients_hook)

    output = model(input_tensor)
    prediction = torch.sigmoid(output).item()

    model.zero_grad()
    output.backward(torch.ones_like(output))

    feature_handle.remove()
    gradient_handle.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap.numpy(), prediction


def visualize_heatmap(image_path, heatmap, prediction, actual_label):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))

    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize(img.size)
    heatmap = heatmap.convert('RGB')

    heatmap = np.array(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.6 + np.array(img)
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f'Actual: {actual_label} - Prediction: {"Real" if prediction > 0.5 else "Fake"} ({prediction:.2f})')
    plt.show()



real_image_paths = ['./dataset/real/54851.jpg','./dataset/real/54852.jpg','./dataset/real/54853.jpg','./dataset/real/54854.jpg','./dataset/real/54855.jpg','./dataset/real/54856.jpg', './dataset/real/69979.jpg', './dataset/real/69987.jpg', './dataset/real/69995.jpg', './dataset/real/69996.jpg', './dataset/real/69997.jpg', './dataset/real/69998.jpg', './dataset/real/69999.jpg']
fake_image_paths = ['./dataset/fake/ZZSLSBZ823.jpg', './dataset/fake/ZZL9HSNP8R.jpg', './dataset/fake/ZZK7DY74LZ.jpg', './dataset/fake/ZZG0G5JZUN.jpg', './dataset/fake/ZYXWKWVG0N.jpg', './dataset/fake/ZXZ7SKVEVM.jpg', './dataset/fake/ZXJ32SOMTT.jpg', './dataset/fake/ZX3M90A4IV.jpg', './dataset/fake/ZXKUXRI00A.jpg', './dataset/fake/ZXFTS9X5OJ.jpg', './dataset/fake/ZVZ8L68C09.jpg', './dataset/fake/ZUZXU25933.jpg', './dataset/fake/ZQH5YUFG5V.jpg', './dataset/fake/ZQ8B3LICA9.jpg', './dataset/fake/ZPGUKG1CJY.jpg', './dataset/fake/ZORO1Z4LZ8.jpg']

# Example usage
for image_path in real_image_paths + fake_image_paths:
    actual_label = 'Real' if 'real' in image_path else 'Fake'
    heatmap, prediction = grad_cam(model, image_path, 'layer4')
    visualize_heatmap(image_path, heatmap, prediction, actual_label)

# Example usage
# for image_path in real_image_paths + fake_image_paths:
#     heatmap, prediction = grad_cam(model, image_path, 'layer4')
#     visualize_heatmap(image_path, heatmap, prediction)

# for image_path in real_image_paths:
#     heatmap = grad_cam(model, image_path, 'layer4')
#     visualize_heatmap(image_path, heatmap)

# for image_path in fake_image_paths:
#     heatmap = grad_cam(model, image_path, 'layer4')
#     visualize_heatmap(image_path, heatmap)