import torch
from torch import nn
from d2l import torch as d2l

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

Lenet = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

X = torch.rand((1, 1, 28, 28), dtype=torch.float32)
for layer in Lenet:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape: \t', X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print(len(train_iter), len(test_iter))

lr, num_epochs = 0.1, 10
d2l.train_ch6(Lenet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())