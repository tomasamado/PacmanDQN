import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model. Takes preprocessed screenshot 
    of the game screen as input and outputs actions to take.

    Attributes:
        input_shape (tuple): Shape of the input tensor.
        num_actions (int): Number of possible actions to take
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(3136, 512) #3136 = size of last flattened conv output #5184 for 100x100
        self.fc2 = nn.Linear(512, num_actions)        

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x