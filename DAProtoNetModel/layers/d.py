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

        output = grad_output.neg() * ctx.alpha #@jinhui 去掉.neg() 可以尝试学习领域特有信息
        # output = grad_output * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    
    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)


        self.domainClass = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/8)*2),
            nn.ReLU(),
            nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/16), 2)
        )

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        logits = self.domainClass(x)
        return logits
