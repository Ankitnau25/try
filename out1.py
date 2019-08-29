import sys
import json
import argparse
import numpy as np
from time import time

# --
# User code
# Note: Depending on how you implement your model, you'll likely have to change the parameters of these
# functions.  They way their shown is just one possble way that the code could be structured.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class ConvRes(nn.Module):
    def __init__(self,in_channel,out_channel,li,scaling_factor):
          super(ConvRes, self).__init__()
          self.conv1=nn.Conv2d(in_channel,16 , kernel_size=3, stride=1, padding=1)
          self.bn1 = nn.BatchNorm2d(16)
          self.relu=nn.ReLU(inplace=True)
          
          ########residual_block1--part1
          self.conv2 = nn.Conv2d(li[0][0], li[0][1], kernel_size=1, stride=1, padding=0)
          
          ########residual_block1--part2
          self.conv3 = nn.Conv2d(li[0][0], li[0][1], kernel_size=3, stride=1, padding=1)
          self.bn3 = nn.BatchNorm2d(li[0][1])
          self.relu = nn.ReLU(inplace=True)
          self.pool1 = nn.MaxPool2d(kernel_size=2)
          #------------
          ########residual_block2--part1
          self.conv4 = nn.Conv2d(li[1][0],li[1][1], kernel_size=1, stride=1, padding=0)
          ########residual_block2--part2
          self.conv5 = nn.Conv2d(li[1][0],li[1][1], kernel_size=3, stride=1, padding=1)
          self.bn4 = nn.BatchNorm2d(li[1][1])
          self.relu = nn.ReLU(inplace=True)
          self.pool2 = nn.MaxPool2d(kernel_size=2)
          ########residual_block2--part1
          self.conv6 = nn.Conv2d(li[2][0],li[2][1], kernel_size=1, stride=1, padding=0)
          ########residual_block2--part2
          self.conv7 = nn.Conv2d(li[2][0],li[2][1], kernel_size=3, stride=1, padding=1)
          self.bn5 = nn.BatchNorm2d(li[2][1])
          self.relu = nn.ReLU(inplace=True)
          self.ln1=nn.Linear(li[2][1],out_channel,bias=False)
          self.sm=nn.Softmax()
    def forward(self, x):
          identity = x
          print(1,x.size())
          out = self.conv1(x)
          print(2,out.size())
          out = self.bn1(out)
          print(3,out.size())
          out_re = self.relu(out)
          out_i =  self.conv2(out_re)
          out_o =  self.conv3(out_re)
          out_o = self.bn3(out_o)
          print(4,out_o.size())
          out=self.relu(out_o) + out_i
          print(5,out.size())
          out_mx=self.pool1(out)
          print(6,out.size())
          out_i =  self.conv4(out_mx)
          print(7,out_i.size())
          out_o =  self.conv5(out_mx)
          out_o = self.bn4(out_o)
          print(8,out_o.size())
          out=self.relu(out_o) + out_i
          print(9,out.size())
          out_mx=self.pool2(out)
          print(10,out.size())
          out_i =  self.conv6(out_mx)
          print(11,out_i.size())
          out_o =  self.conv7(out_mx)
          out_o = self.bn5(out_o)
          print(12,out_o.size())
          out=self.relu(out_o) + out_i
          print(13,out.size())
          out_mx=nn.MaxPool2d(kernel_size=out.size()[2:])(out).view(128,128)
          print("14",out_mx.size())
          out=self.ln1(out_mx)
          print("linear",out.size())
          out=scaling_factor*out
          print("scaling",out.size())
#           out=torch.sum(out)
#           print("sum",out)
          out=self.sm(out)
          print("sm,",out.size())
    
class MyCustomDataset(Dataset):
    def __init__(self, x,y,dtype):
        self.x=torch.tensor(torch.from_numpy(x),dtype=dtype)
        self.y=torch.tensor(torch.from_numpy(y),dtype=dtype)
        self.dtype=dtype
        self.data_len=len(x)
    def __getitem__(self, index):
        # stuff
        img=self.x[index]
        label=self.y[index]
        return (img, label)

    def __len__(self):
        return self.data_len

def make_model(input_channels, output_classes, residual_block_sizes, scale_alpha):
    # ... your code here ...
    model=ConvRes(input_channels,output_classes,residual_block_sizes,scale_alpha)
    return model


def make_train_dataloader(X, y, batch_size, shuffle):
    # ... your code here ...
    batch_size=batch_size
    dtype = torch.float
    train_dataset=MyCustomDataset(X_train,Y_train,dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader


def make_test_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    batch_size=batch_size
    dtype = torch.float
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader


def train_one_epoch(optimizer,criterion,epoch,model, train_loader):
    # ... your code here ...
    device=torch.device("cpu")
    model.train()
    train_loss = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = augment_data(data, target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)    
    return model


def predict(criterion,model,test_loader):
    # ... your code here ...
    model.eval()
    test_loss = 0
    pred=[]
    device=torch.device("cpu")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred.append(output)
            test_loss += criterion(output, target).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
#     print(test_loss)
#     print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
 
    return torch.stack(pred)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # --
    # IO
    
    # X_train: tensor of shape (number of train observations, number of image channels, image height, image width)
    # X_test:  tensor of shape (number of train observations, number of image channels, image height, image width)
    # y_train: vector of [0, 1] class labels for each train image
    # y_test:  vector of [0, 1] class labels for each test image (don't look at these to make predictions!)
    
    X_train = np.load('data/cifar2/X_train.npy')
    X_test  = np.load('data/cifar2/X_test.npy')
    y_train = np.load('data/cifar2/y_train.npy')
    y_test  = np.load('data/cifar2/y_test.npy')
    
    # --
    # Define model
    
    model = make_model(
        input_channels=3,
        output_classes=2,
        residual_block_sizes=[
            (16, 32),
            (32, 64),
            (64, 128),
        ],
        scale_alpha=0.125
    )
    
    # --
    # Train
    lr=args.lr,
    momentum=args.momentum,
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=momentum)
    t = time()
    for epoch in range(args.num_epochs):
        
        # Train
        model = train_one_epoch(optimizer,criterion,epoch,
            model=model,
            dataloader=make_train_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
            
        )
        
        # Evaluate
        preds = predict(criterion,
            model=model,
            dataloader=make_test_dataloader(X_test, batch_size=args.batch_size, shuffle=False)
            
        )
        
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X_test.shape[0]
        
        test_acc = (preds == y_test.squeeze()).mean()
        
        print(json.dumps({
            "epoch"    : int(epoch),
            "test_acc" : test_acc,
            "time"     : time() - t
        }))
        sys.stdout.flush()
        
    elapsed = time() - t
    print('elapsed', elapsed, file=sys.stderr)
    
    # --
    # Save results
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/preds', preds, fmt='%d')
    open('results/elapsed', 'w').write(str(elapsed))
