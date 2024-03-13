import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet18
import os

# Define the transformations to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = resnet18(num_classes=2)
model.load_state_dict(torch.load('deepfake_classifier.pth'))
model.eval()

# Specify the paths to the "real" and "fake" datasets
real_dataset_path = 'dataset/real'
fake_dataset_path = 'dataset/fake'

# Get the first image filename from each dataset
real_image_filename = os.listdir(real_dataset_path)[1600]
fake_image_filename = os.listdir(fake_dataset_path)[1600]

# Construct the full paths to the first images
real_image_path = os.path.join(real_dataset_path, real_image_filename)
fake_image_path = os.path.join(fake_dataset_path, fake_image_filename)

# Function to classify an image
def classify_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to RGB format
    image = image.convert('RGB')
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Print the predicted class
    if predicted.item() == 0:
        print(f"The image '{image_path}' is classified as real.")
    else:
        print(f"The image '{image_path}' is classified as fake.")

# Classify the first image from the "real" dataset
classify_image(real_image_path)

# Classify the first image from the "fake" dataset
classify_image(fake_image_path)