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
    def __init__(self, sentence_encoder, hidden_size, graphFeature_size=100, dot=False):
        framework.FewShotREModel.__init__(self, sentence_encoder, hidden_size)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc = nn.Linear(hidden_size*2, hidden_size)

        self.featuretransfer = nn.Sequential(nn.Linear(hidden_size, hidden_size*2),
                                         nn.Linear(hidden_size*2, hidden_size)
                                         )

        # TODO jinhui
        # graph feature map
        # encoder
        self.g1 = nn.Linear(graphFeature_size, hidden_size)
        self.g1_drop = nn.Dropout()
        # self.fc2 = nn.Linear(hidden_size * 2, hidden_size)


        # graph feature decoder
        self.d1 = nn.Linear(hidden_size, 100)
        self.d2 = nn.Linear(100, graphFeature_size)
        self.d_drop = nn.Dropout()







    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    def getGraphFeature(self,support, query):
        support_graphFeature, query_graphFeature = support["graphFeature"], query["graphFeature"]
        return support_graphFeature, query_graphFeature

    def graphFeatureRecon(self, graphFeature):

        graphFeature_maps = self.g1(graphFeature)  # 也没有inplace operator
        # query_graphFeature_maps = self.g1(query_graphFeature)

        graphFeature_ = self.d1(graphFeature_maps)
        graphFeature_ = F.relu(graphFeature_)
        graphFeature_ = self.d_drop(graphFeature_)
        # z = torch.sigmoid(self.d2(z))
        graphFeature_ = self.d2(graphFeature_)
        return graphFeature_

    def loss_recon(self, recon_x, x):
        dim = x.size(1)
        MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
        return MSE


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

        # # graph feature maps
        # support_graphFeature, query_graphFeature = self.getGraphFeature(support, query)
        # support_graphFeature_maps = self.g1(support_graphFeature)# 也没有inplace operator
        # query_graphFeature_maps = self.g1(query_graphFeature)
        #
        #
        # sp_support_emb = torch.cat([sp_support_emb, support_graphFeature_maps], axis=1)
        # sp_query_emb = torch.cat([sp_query_emb, query_graphFeature_maps], axis=1)
        # sp_support_feat = self.fc2(sp_support_emb) # 按理说犯了inplace错误呀
        # sp_query_feat = self.fc2(sp_query_emb)
        #
        # # end @jinhui


        ## 单单使用graphfeature 作为sp_feature
        # graph feature maps
        support_graphFeature, query_graphFeature = self.getGraphFeature(support, query)
        sp_support_emb = self.g1(support_graphFeature)# 也没有inplace operator
        sp_query_emb = self.g1(query_graphFeature)

        # end @jinhui

        ## 只使用in_encoder

        # support_emb = self.featuretransfer(in_support_emb)
        # query_emb = self.featuretransfer(in_query_emb)
        """
            学习领域不变特征  Learn Domain Invariant Feature
        """
        support_emb = torch.cat([in_support_emb, sp_support_emb], axis=1)  # (B*N*K, 2D)
        query_emb = torch.cat([in_query_emb, sp_query_emb], axis=1)  # (B*Q*N, 2D)
        support_emb = self.fc(support_emb)
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

    def forwad_postBERT_newGraph(self, support, query, N, K, total_Q):
        in_support_emb, sp_support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        in_query_emb, sp_query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = in_support_emb.size(-1)

        support_emb = torch.cat([in_support_emb, sp_support_emb], axis=1)  # (B*N*K, 2D)
        query_emb = torch.cat([in_query_emb, sp_query_emb], axis=1)  # (B*Q*N, 2D)
        support_emb = self.fc(support_emb)
        query_emb = self.fc(query_emb)

        ## 单单使用graphfeature 作为sp_feature
        # graph feature maps
        support_graphFeature, query_graphFeature = self.getGraphFeature(support, query)
        sp_support_emb = self.g1(support_graphFeature)  # 也没有inplace operator
        sp_query_emb = self.g1(query_graphFeature)


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


