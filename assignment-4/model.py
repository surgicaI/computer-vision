import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        kernel_size = 5
        input_channels = 3
        output_channels_conv_layer_1 = 16
        output_channels_conv_layer_2 = 128
        num_units_hidden_layer_1 = output_channels_conv_layer_2 * 5 * 5
        num_units_hidden_layer_2 = 64
        self.conv1 = nn.Conv2d(input_channels, output_channels_conv_layer_1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(output_channels_conv_layer_1, output_channels_conv_layer_2, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(num_units_hidden_layer_1, num_units_hidden_layer_2)
        self.fc2 = nn.Linear(num_units_hidden_layer_2, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
