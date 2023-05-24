import torch.nn as nn

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class RiceDetector(nn.Module):
    
    def __init__(self):
        # build a CNN network
        
        super(RiceDetector, self).__init__()
        # (3, 250, 250)
        self.conv_1 = nn.Conv2d(3, 64, (3, 3))      # (64, 248, 248)
        self.pool_1 = nn.MaxPool2d((2, 2))          # (64, 124, 124)
        self.conv_2 = nn.Conv2d(64, 128, (5, 5))    # (128, 120, 120)
        self.pool_2 = nn.MaxPool2d((2, 2))          # (128, 60, 60)
        self.conv_3 = nn.Conv2d(128, 256, (5, 5))   # (256, 56, 56)
        self.pool_3 = nn.MaxPool2d((2, 2))          # (256, 28, 28)
        self.conv_4 = nn.Conv2d(256, 128, (3, 3))   # (128, 26, 26)
        self.pool_4 = nn.MaxPool2d((2, 2))          # (128, 13, 13)
        
        # classifier
        self.fc_1 = nn.Linear(128*13*13, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 5)
        
        
        # activation
        self.relu = nn.ReLU()
        
        # flatten layer
        self.flatten = nn.Flatten()
        
        # BatchNorm layer
        self.batch_norm_1 = nn.BatchNorm2d(3)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.batch_norm_4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # feature extraction
        x = self.batch_norm_1(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool_1(x)
        
        x = self.batch_norm_2(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool_2(x)
        
        x = self.batch_norm_3(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.pool_3(x)
        
        x = self.batch_norm_4(x)
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.pool_4(x)
        
        # Flatten
        x = self.flatten(x)
        
        # classifier
        x = self.fc_1(x)
        x = self.relu(x)
        
        x = self.fc_2(x)
        x = self.relu(x)
        
        logits = self.fc_3(x)
        
        # Softmax
        prob = nn.Softmax(logits).dim
        
        return prob
        