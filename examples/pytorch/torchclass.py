def shoe_model_list():
    import timm
    from pprint import pprint
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
#shoe_model_list()
#exit()

#model_name = 'mobilenetv3_large_100'
#model_name = 'mobilenetv2_100'
#model_name = 'convnext_base'
#model_name = 'resnet50'
model_name = 'vit_base_patch8_224'

import timm
model = timm.create_model(model_name, pretrained=True)
model.eval()

import torch
img = torch.randn(1, 3, 224, 224, device=torch.device('cpu'), dtype=torch.float32)
#img = torch.tensor(img)
torch.onnx.export(model, img, model_name+'.onnx', export_params=True, verbose=False,
    do_constant_folding=True,
    input_names=['Input'],
    output_names=['Output'],
    opset_version=11,
    dynamic_axes={"Input":{0:"batch_size"},"Output":{0:"batch_size"}})

import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)
filename = "dog.jpg"
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension

from torchsummary import summary
summary(model,tensor,device="cpu")

import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])

# Get imagenet class mappings
#url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
#urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
