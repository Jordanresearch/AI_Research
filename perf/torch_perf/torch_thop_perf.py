import torch
import torchvision
from thop import profile
 
# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)
 
dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
