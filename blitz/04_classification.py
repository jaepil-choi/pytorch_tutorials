# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Training a Classifier

#%%
import torch, torchvision
import torchvision.transforms as transforms
# %%
## 1. Loading and normalizing CIFAR10

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    ]
)
print(transform)
print(type(transform))
# %%
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

trainset
# %%
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=6
)

trainloader
# %%
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform,
)

testset
#%%
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=4,
    shuffle=False,
    num_workers=6
)

testloader
# %%
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%
### Check some images

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
#%%
print(images.shape)
print(labels.shape)
# %%
## 2. Define a CNN

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16 * 5 * 5)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x

net = Net()
net.to(device)
net
# %%
## Define a Loss function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(criterion)
print(optimizer)
# %%
## 3. Train the network

for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(trainloader, start=0):
        # get the input; data is a list of [inputs, labels]
        inputs, labels = data 
        inputs, labels = inputs.to(device), labels.to(device)
        ## inputs: [4, 3, 32, 32] labels: [4]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[epoch: {epoch + 1}, iter: {i+1}] loss = {runninig_loss / 2000}')
            
            runninig_loss = 0.0
# %%
### Save the trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
# %%

## 4. Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%
### Load saved model
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
outputs
#%%
### Calculate accuracy

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}')
#%%
### Calculate accuracy for all classes
##################여기 하다맘. 

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, dim=1)
        c = (predicted == labels).squeeze() ### 이건 뭔지?? tensor squeeze 하면 어떻게 되나????
        ### 이렇게 for 문 안에 뭐 나왔을 때 빠르게 디버깅하려면 어떻게 해야 하는지??? 계속 print 찍지 않고 한 번만 보고싶은데. 

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

