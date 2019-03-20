import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 25
num_classes = 10
batch_size = 32
num_workers=4
learning_rate = 0.001
weight_decay = 0.0001

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Define transformations for the test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                           train=True, 
                                           transform=train_transformations,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                          train=False, 
                                          transform=test_transformations)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=num_workers)

"""
CIFAR10 input image: 3 * 32 * 32
"""
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU())
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(1*1*128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        out = self.avgpool(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class ConvNet_with_bn(nn.Module):
    """
    Add batch normalization layer right after each convolutional layers.
    """
    def __init__(self, num_classes=10):
        super(ConvNet_with_bn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU())
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(1*1*128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        out = self.avgpool(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class ConvNet_deep(nn.Module):
    """
    Using smaller filter and deeper layers.
    """
    def __init__(self, num_classes=10):
        super(ConvNet_deep, self).__init__()
        self.layer1 = nn.Sequential( # 3*32*32 -> 64*16*16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential( # 64*16*16 -> 128*8*8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential( # 128*8*8 -> 256*4*4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential( # 256*4*4 -> 512*4*4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU())
        self.avgpool = nn.AvgPool2d(kernel_size=4) # 512*4*4 -> 512*1*1
        self.fc = nn.Linear(1*1*512, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def model_train(model=None):
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_lst = []
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        epoch_loss = total_loss / len(train_loader.dataset)
        loss_lst.append(epoch_loss)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    loss_lst = [l for l in loss_lst]

    # Draw loss-epoch graph.
    def draw_loss():
        ts = time.gmtime()
        plt.title("Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,num_epochs+1),loss_lst)
        plt.xticks(np.arange(1, num_epochs+1, 1.0))
        plt.legend()
        plt.savefig('loss-epoch{}.png'.format(time.strftime("%Y-%m-%d %H:%M:%S", ts)))

    draw_loss()

    # Visualize conv filter
    def vis_kernels():
        kernels = model.layer1[0].weight.detach()
        print(kernels.shape)
        fig, axarr = plt.subplots(4, 16, figsize=(15, 15))
        for x in range(4):
            for y in range(16):
                kernel_id = x * 4 + y
                kernel = kernels[kernel_id]
                # print(kernel.shape)
                axarr[x, y].imshow(transforms.ToPILImage()(kernel), interpolation="bicubic")

# model = ConvNet(num_classes).to(device)
# model_train(model)

# model_bn = ConvNet_with_bn(num_classes).to(device) # model with batch normalization layer between each hidden layers. 
# model_train(model_bn)

model_deep = ConvNet_deep(num_classes).to(device) 
model_train(model_deep)