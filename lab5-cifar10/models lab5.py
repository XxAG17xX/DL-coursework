import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)

class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.pool =nn.MaxPool2d(2, 2)
        self.drop =nn.Dropout(0.4)

        # --Conv blocks
        self.conv1 =nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   =nn.BatchNorm2d(64)
        self.conv2 =nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   =nn.BatchNorm2d(64)
        self.conv3 =nn.Conv2d(64, 128,kernel_size=3, padding=1)
        self.bn3   =nn.BatchNorm2d(128)
        self.conv4 =nn.Conv2d(128, 128,kernel_size=3, padding=1)
        self.bn4   =nn.BatchNorm2d(128)
        self.conv5 =nn.Conv2d(128, 256,kernel_size=3, padding=1)
        self.bn5   =nn.BatchNorm2d(256)
        self.conv6 =nn.Conv2d(256, 256,kernel_size=3, padding=1)
        self.bn6   =nn.BatchNorm2d(256)
        #Classifier 
        #after three pooling operations(32→16→8→4)
        self.fc1 =nn.Linear(256 * 4 * 4, 512)
        self.fc2 =nn.Linear(512,num_classes)

    def forward(self,x):
        # block 1
        x =F.leaky_relu(self.bn1(self.conv1(x)))
        x =self.pool(F.leaky_relu(self.bn2(self.conv2(x))))

        # block 2
        x =F.leaky_relu(self.bn3(self.conv3(x)))
        x =self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        # block 3
        x =F.leaky_relu(self.bn5(self.conv5(x)))
        x =self.pool(F.leaky_relu(self.bn6(self.conv6(x))))

        x =x.view(x.size(0),-1)
        x =self.drop(F.leaky_relu(self.fc1(x)))
        x =self.fc2(x)
        return x