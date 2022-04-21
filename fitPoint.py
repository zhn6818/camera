import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
 
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


class LinearRegression(nn.Module):
  def __init__(self):
    super(LinearRegression,self).__init__()
    self.linear = nn.Linear(2,100)
    self.linear1 = nn.Linear(100,100)
    self.linear1_1 = nn.Linear(100,100)
    self.linear2 = nn.Linear(100,2)
  def forward(self,x):
    x = self.linear(x)
    for _ in range(3):
      x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(x)
    x = nn.functional.relu(self.linear1_1(x))
    x = self.linear2(x)
    return x

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model = LinearRegression()
model.apply(weight_init)


criterion = nn.MSELoss()
import torch.optim as optim
optimizer = optim.SGD(model.parameters(),lr=1e-3)

x_train_list = [[2,2], [2,3], [3,2],[3,3]]
y_train_list = [[0,0], [0,0.2], [0.2, 0], [0.2, 0.2]]

x_train = np.array(x_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

if __name__ == '__main__':
    num_epochs = 80000
    for epoch in range(num_epochs):
        input = torch.autograd.Variable(x_train)
        target = torch.autograd.Variable(y_train)
        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad() #清除上一梯度
        loss.backward() #梯度计算
        optimizer.step()#梯度优化
        # if (epoch+1) % 4 == 0:
        print('Epoch[{}/{}],loss:{:.4f}'.format(epoch, num_epochs,loss.item()))
        torch.save(model.state_dict(), "./checkPoint.pth")


