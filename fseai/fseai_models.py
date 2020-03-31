import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        # Hidden layer 1 to HL2 linear transformation
        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        # HL2 to output linear transformation
        self.layer3 = nn.Linear(hidden_size[1], output_size)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # HL1 with relu activation
        out = self.relu(self.layer1(x))
        # HL2 with relu activation
        out = self.relu(self.layer2(out))
        # Output layer with LogSoftmax activation
        out = self.softmax(self.layer3(out))
        return out


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
