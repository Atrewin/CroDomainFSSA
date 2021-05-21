import sys

sys.path.append('..')
from DAProtoNetModel.layers import framework
from DAProtoNetModel.layers.network.functions import ReverseLayerF
import torch
from torch import autograd, optim, nn
from DAProtoNetModel.layers.d import Discriminator
from DAProtoNetModel.layers.sen_d import Sen_Discriminator, Sen_Discriminator_sp


class DAPostBertNet(framework.FewShotREModel):
    # 类似于
    def __init__(self, sentence_encoder, hidden_size, sentiment_classifier=None, discrimator=None,  dot=False):
        framework.FewShotREModel.__init__(self, sentence_encoder, hidden_size)
        # self.fc = nn.Linear(hidden_size, hidden_size)

        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        # TODO jinhui
        # graph feature map
        self.sentimentClassifier = nn.Sequential(nn.Dropout(0.2),
                                                 nn.Linear(hidden_size, 2)
                                                 )
        # self.sentimentClassifier =  nn.Sequential(
        #     nn.Linear(hidden_size, int(hidden_size/8)*2),
        #     nn.ReLU(),
        #     nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
        #     nn.ReLU(),
        #     nn.Linear(int(hidden_size/16), 2)
        # )

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
        in_support_emb= self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        logits = self.sentimentClassifier(in_support_emb)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred


