from dinov2.models.vision_transformer import vit_base
import torch
from PIL import Image
import requests
from torchvision import transforms 
model = vit_base()

for name, param in model.named_parameters():
    print(name, param.shape)

state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth', 
                                                map_location='cpu')

model.load_state_dictd(state_dict)

# Load Image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess Image
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
])

pixel_values = transformations(image).unsqeeze(0)

output = model(pixel_values)

print("Outputs", output)



