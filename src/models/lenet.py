import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=5 * 5 * 50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)

        x = x.view(-1, 5 * 5 * 50)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = self.forward(x)
            pred = torch.argmax(x, dim=1)

            return pred.item()
