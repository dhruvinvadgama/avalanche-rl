import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


batch_size = 64

train_MNIST_loader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
test_MNIST_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

num_classes = 10
input_size = 28 * 28
hidden_size = 512
drop_rate = 0.5


class MLP(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, drop_rate):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.flat = nn.Flatten()
        self.activation = nn.ReLU()
        self.drop = nn.Dropout()
        self.linear_layer_1 = nn.Linear(28 * 28, hidden_size)
        self.linear_layer_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = self.flat(x)
        # x = x.view(x.size(0), self._input_size)
        x = self.linear_layer_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear_layer_2(x)
        x = self.activation(x)
        x = self.drop(x)
        return x


model = MLP(num_classes, input_size, hidden_size, drop_rate)

lr = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
epochs = 2

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for input_img, target in train_MNIST_loader:
