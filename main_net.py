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
from sklearn.preprocessing import OneHotEncoder

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
          self.scaling_factor=scaling_factor
          self.sm=nn.Softmax()
    def forward(self, x):
          identity = x
          out = self.conv1(x)
          out = self.bn1(out)
          out_re = self.relu(out)
          out_i =  self.conv2(out_re)
          out_o =  self.conv3(out_re)
          out_o = self.bn3(out_o)
          out=self.relu(out_o) + out_i
          out_mx=self.pool1(out)
          out_i =  self.conv4(out_mx)
          out_o =  self.conv5(out_mx)
          out_o = self.bn4(out_o)
          out=self.relu(out_o) + out_i
          out_mx=self.pool2(out)
          out_i =  self.conv6(out_mx)
          out_o =  self.conv7(out_mx)
          out_o = self.bn5(out_o)
          out=self.relu(out_o) + out_i
          dim1=out.size()[0]
          dim2=out.size()[1]
          out_mx=nn.MaxPool2d(kernel_size=out.size()[2:])(out).view(dim1,dim2)
          out=self.ln1(out_mx)
          out=self.scaling_factor*out
          out=self.sm(out)
          return out
      
class MyCustomDataset(Dataset):
    def __init__(self, x,y,dtype):
        self.x=torch.tensor(torch.from_numpy(x),dtype=torch.float)
        self.y=torch.tensor(torch.from_numpy(y),dtype=torch.float)
        self.dtype=dtype
        self.data_len=len(x)
    def __getitem__(self, index):
        img=self.x[index]
        label=self.y[index]
        return (img, label)

    def __len__(self):
        return self.data_len
class Cat_Cross_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Cat_Cross_Loss,self).__init__()
        
    def forward(self,x,y):
        
        ent = 0
        for i in range(2):
            val=torch.mean(y[:,i]*torch.log(x[:,i]+(1e-12)))
            ent += val
        return -1*ent

def make_model(input_channels, output_classes, residual_block_sizes, scale_alpha):
    # ... your code here ...
    model=ConvRes(input_channels,output_classes,residual_block_sizes,scale_alpha)
    return model


def make_train_dataloader(X, Y_train, batch_size, shuffle):
    # ... your code here ...
    batch_size=batch_size
    dtype = torch.float
    train_dataset=MyCustomDataset(X_train,Y_train,dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    return train_loader


def make_test_dataloader(X_test,Y_test, batch_size, shuffle):
    # ... your code here ...
    batch_size=batch_size
    dtype = torch.float
    test_dataset=MyCustomDataset(X_test,Y_test,dtype)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
    return test_loader


def train_one_epoch(optimizer,criterion,epoch,model,train_loader):
    # ... your code here ...
    device=torch.device("cpu")
    model.train()
    train_loss = 0 
   
    for batch_idx, (data, target) in enumerate(train_loader):
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
            out = 0.5<output
            _, argmax = out.max(-1)
            pred.append(argmax)
              
            test_loss += criterion(output, target).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
 
    return torch.cat(pred).numpy()

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
#     args = parse_args()
    
    # --
    # IO
    
    # X_train: tensor of shape (number of train observations, number of image channels, image height, image width)
    # X_test:  tensor of shape (number of train observations, number of image channels, image height, image width)
    # y_train: vector of [0, 1] class labels for each train image
    # y_test:  vector of [0, 1] class labels for each test image (don't look at these to make predictions!)
    enc = OneHotEncoder(handle_unknown='ignore')

    X_train = np.load('data/cifar2/X_train.npy')
    X_test  = np.load('data/cifar2/X_test.npy')
    y_train = np.load('data/cifar2/y_train.npy')
    y_test  = np.load('data/cifar2/y_test.npy')
    y_test_one = y_test.copy()
    enc.fit(y_train.reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test  = enc.transform(y_test.reshape(-1,1)).toarray()
    print(X_train.shape,y_train.shape)
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
    lr=0.1
    momentum=0.9
    criterion = Cat_Cross_Loss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=momentum)
    t = time()
    for epoch in range(5):
        
        # Train
        model = train_one_epoch(optimizer,criterion,epoch,model=model,train_loader=make_train_dataloader(X_train, y_train, batch_size=128, shuffle=True)
            
        )
        
        # Evaluate
        preds = predict(criterion, model=model,test_loader=make_test_dataloader(X_test,y_test, batch_size=128, shuffle=False)
            
        )
        
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X_test.shape[0]
        test_acc = (preds == y_test_one.squeeze()).mean()
        
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