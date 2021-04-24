import sys
sys.path.append('..')
from torch import nn
# 梯度反转层
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        output = grad_output * ctx.alpha #@jinhui 去掉.neg() 可以尝试学习领域特有信息
        # output = grad_output * ctx.alpha
        return output, None

class Sen_Discriminator(nn.Module):

    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.sentimentClass = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/8)*2),
            nn.ReLU(),
            nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/16), hidden_size),
            nn.Linear(hidden_size, int(hidden_size / 8) * 2),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 8) * 2, int(hidden_size / 16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 16), 2)
        )

        # self.hidden_size = hidden_size
        # self.num_labels = num_labels
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.relu1 = nn.ReLU()
        # self.drop = nn.Dropout()
        # self.fc2 = nn.Linear(hidden_size, 2)# 这个学不起来的，因为drop()的缘故


    def forward(self, x,alpha=1):
        x = ReverseLayerF.apply(x,alpha)
        logits = self.sentimentClass(x)

        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.drop(x)
        # logits = self.fc2(x)
        return logits


class Sen_Discriminator_sp(nn.Module):

    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.sentimentClass = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/8)*2),
            nn.ReLU(),
            nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/16), 2)
        )

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, 2)


    def forward(self, x, alpha=1):
        x = ReverseLayerF.apply(x, alpha)
        logits = self.sentimentClass(x)

        return logits