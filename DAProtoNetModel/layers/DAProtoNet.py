import sys

sys.path.append('..')
from DAProtoNetModel.layers import framework
from DAProtoNetModel.layers.network.functions import ReverseLayerF
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class DAProtoNet(framework.FewShotREModel):
    # 类似于
    def __init__(self, sentence_encoder, hidden_size, dot=False):
        framework.FewShotREModel.__init__(self, sentence_encoder, hidden_size)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        in_support_emb, sp_support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        in_query_emb, sp_query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = in_support_emb.size(-1)

        """
            学习领域不变特征  Learn Domain Invariant Feature
        """
        support_emb = torch.cat([in_support_emb, sp_support_emb], axis=1)  # (B*N*K, 2D)
        support_emb = self.fc(support_emb)
        query_emb = torch.cat([in_query_emb, sp_query_emb], axis=1)  # (B*Q*N, 2D)
        query_emb = self.fc(query_emb)

        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

        # Prototypical Networks
        # Ignore NA policy
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred

