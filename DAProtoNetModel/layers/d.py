import sys
sys.path.append('..')
from torch import autograd, optim, nn
# 梯度反转层
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=0.5):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        output = grad_output * ctx.alpha #@jinhui 为了尝试学习领域特有信息
        return output, None

class Discriminator(nn.Module):
    
    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, 2)

        self.ReverseLayerF = ReverseLayerF()

    def forward(self, x):
        x = ReverseLayerF.apply(x, alpha)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits
