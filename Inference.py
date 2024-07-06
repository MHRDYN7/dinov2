from dinov2.models.vision_transformer import vit_base
import torch
from PIL import Image
import requests
from torchvision import transforms 
model = vit_base(img_size = 518, patch_size = 14, init_values = 1.0, block_chunks = 0, num_register_tokens=4)

#for name, param in model.named_parameters():
#    print(name, param.shape)

hub_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

model.load_state_dict(hub_model.state_dict())

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

pixel_values = transformations(image).unsqueeze(0)

outputs = model.forward_features(pixel_values)

for k, v in outputs.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
  else:
    print(k, v)

#print("The Outputs have a shape", outputs)



