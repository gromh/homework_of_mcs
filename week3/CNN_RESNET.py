import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import  DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# %matplotlib inline
#超参数
epochs = 25
lr = 0.0001
batch_size = 64

#数据加载
train_data = datasets.CIFAR10(root='/data', train=True,
                              transform=transforms.ToTensor(),
                              download=True)
test_data = datasets.CIFAR10(root = '/data',train=False,
                             transform=transforms.ToTensor(),
                             download=True)

#看一眼
# temp = train_data[1][0].numpy()
# print(temp.shape)
#(3, 32, 32) 3个通道（rgb）， 每个图片大小32 * 32

# temp = temp.transpose(1, 2, 0)
# print(temp.shape)
# plt.inshow(temp)
# plt.show()


train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset=test_data, batch_size = batch_size)

model = torchvision.models.resnet50(pretrained=True)
# model = torch.load('CNN')
criterion = nn.CrossEntropyLoss()
opt = opt.Adam(model.parameters(), lr=lr)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, label)

        opt.zero_grad()
        loss.backward()
        opt.step()
    print('epoch:{}, loss:{:.4f}'.format(epoch+1, loss.item()))

torch.save(model, 'CNN_pretrain')
print('CNN_pretrain is saved')

model.eval()
correct, total = 0, 0
for data in test_loader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = total + labels.size(0)
    correct = correct + (predicted==labels).sum().item()

print('准确率{:4f}%'.format(100.0*correct/total))
# 没有预训练 lr = 0.001 准确率70.600000%
# 加入预训练 lr = 0.0001 准确率84.530000%