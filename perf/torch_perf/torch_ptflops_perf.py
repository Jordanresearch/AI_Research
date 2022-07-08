import torchvision
from ptflops import get_model_complexity_info
 
model = torchvision.models.alexnet(pretrained=False)
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)
 
