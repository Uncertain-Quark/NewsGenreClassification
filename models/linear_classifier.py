import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.linear(x)
        return x

if __name__ == '__main__':
    model = LinearClassifier(768)
    print(model)