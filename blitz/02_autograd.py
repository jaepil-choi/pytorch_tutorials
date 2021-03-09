# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# A Gentle Intro to torch.autograd
#%%
import torch, torchvision
from torch import nn, optim
# %%

## Usage in PyTorch

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

#%%
prediction = model(data)
prediction.shape
# %%
loss = (prediction - labels).sum()
loss.backward()
# %%
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()
# %%

## Differentiation in Autograd
a = torch.tensor([2., 3.], requires_grad=True) # every operation is tracked
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2 # a, b: parameter Q: error
# %%
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
# %%
print(a.grad)
print(b.grad)
# %%
### Exclusion from the DAG(Directed Acyclic Graph)

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)
#%%
a = x + y
a.requires_grad # False
# %%
b = x + z
b.requires_grad # True # 하나라도 requires_grad일 경우
# %%
### DAG is used to finetune a pretrained network
# gradient를 계산하지 않는 parameter를 frozen parameter라 부른다. 
# finetuning 시 대부분 classifier layer만 수정하고 나머지 모델은 freeze시킨다. 


model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters(): # Freeze all parameters in the network
    param.requires_grad = False
#%%
model.fc
# %%
model.fc = nn.Linear(512, 10) # Now it is unfrozen by the default. Only this layer will be trained.
model.fc
# %%
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
# %%
