from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

trained_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

# Load the pre-trained ResNet model
class CarDamageClassifierResNet(nn.Module):
    def __init__(self, num_of_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True            
            
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_of_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    # (32, 3, 224, 224) our model accept data in batches of 32. now, use unsqueeze to add new dimension
    image_tensor = image_transforms(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        trained_model = CarDamageClassifierResNet()
        trained_model.load_state_dict(torch.load("./model/car_damage_classification.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor) # [[14, 16, 10, 9, 7, 20]]
        _,predicted_class = torch.max(output, 1) # _ will be value and predicted_class will be index
        print(f"_: {_} | Predicted Class: {predicted_class}")
        return class_names[predicted_class.item()]




    