import torch as t
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.linear = nn.Linear(input_dim, 1).to(self.device)
        self.available_num_options = [2]

    def forward(self, x):
        x = x.to(t.float32)
        return t.sigmoid(self.linear(x)).squeeze(1)

    def get_weight(self):
        return self.linear.weight.squeeze(0)
    
    def get_bias(self):
        return self.linear.bias.item()
    
    def set_weight(self, weight):
        with t.no_grad():
            self.linear.weight.copy_(weight.to(self.device).clone())

    def set_bias(self, bias):
        with t.no_grad():
            self.linear.bias.copy_(t.tensor([bias]).to(self.device))


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.linear = nn.Linear(input_dim, n_classes).to(self.device)
        self.n_classes = n_classes
        self.available_num_options = list(range(2, n_classes + 1))

    def forward(self, x):
        x = x.to(self.device)
        x = x.to(t.float32)

        logits = self.linear(x)  # (B, num_classes), float32
        return t.softmax(logits, dim=1)  # (B, num_classes)

    def get_weight(self):
        # Returns weight of shape (num_classes, input_dim)
        return self.linear.weight.detach()

    def get_bias(self):
        # Returns bias of shape (num_classes,)
        return self.linear.bias.detach()

    def set_weight(self, weight):
        weight = weight.to(self.device, dtype=t.float32)
        with t.no_grad():
            self.linear.weight.copy_(weight)

    def set_bias(self, bias):
        bias = t.tensor(bias, device=self.device, dtype=t.float32)
        with t.no_grad():
            self.linear.bias.copy_(bias)
