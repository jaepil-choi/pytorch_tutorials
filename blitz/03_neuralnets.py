# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

# Neural Networks
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%

## Define the Network

input = torch.randn(1, 1, 32, 32) # nSamples, nChannels, Height, Width 
# 무조건 0번째 batch dimension이 있어야 한다. 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # torch.nn only supports mini-batches. 

        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3) # 1 input image channel, 6 output channels, 3*3 square convolution
        self.conv2 = nn.Conv2d(6, 16, 3)

        # Affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 is image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square, you can only specify a single number
        print(x.shape)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    # def backward(): # backward() is automatically defined when forward() is defined
        # pass

    def num_flat_features(self, x):
        size = x.size()[1:] # All dimensions except the batch dimension
        num_features = 1

        for s in size:
            num_features *= s
        
        return num_features

net = Net()
print(net)
# %%

params = list(net.parameters())
print(params[0].size()) # conv1's weight
len(params)
# %%

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(f'Out Size: {out.size()}')
out
# %%

## Loss function

output = net(input)
target = torch.randn(10)
target = target.view(1, -1) # target should be the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
loss

# %%
loss.grad_fn
# %%
## Backprop

net.zero_grad() # zeros the gradient buffers of all parameters
print(net.conv1.bias.grad) # grad before backward 

# %%
loss.backward()
# %%
print(net.conv1.bias.grad) # grad after backward
# %%

## Update the weights

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
# %%
