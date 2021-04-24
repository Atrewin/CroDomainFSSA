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

        self.sentimentClassifier =  nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/8)*2),
            nn.ReLU(),
            nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/16), 2)
        )

        # self.sentimentClassifier = Sen_Discriminator(hidden_size, 2)#这个好像导致模型无法优化， 是因为内部有太多的drop了

        # self.discrimator = Discriminator(hidden_size, 2)

        #TODO 对抗训练的
        # 情感分析

        # 领域区分






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
    # def getGraphFeatureRecon(self):
    #     return self.reconGraphFeature

    def loss_recon(self, recon_x, x):
        dim = x.size(1)
        # MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
                        # CEL = F.cross_entropy(recon_x, x.view(-1, dim), reduction='mean')# 因为recon_x是小数，并不是分类结果，所以不可以使用cross_entropy
                        # CEL = self.CrossEntropyLoss(recon_x, x)
        cosine = torch.cosine_similarity(recon_x, x.view(-1, dim), dim=1)
        CSD = torch.mean(cosine, 0)
        return CSD


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

    def forwardWithRecon(self, support, query, N, K, total_Q):
        in_support_emb, sp_support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        in_query_emb, sp_query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = in_support_emb.size(-1)

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

        # 回购graph feature
        support_graphFeature, query_graphFeature = sp_support_emb, sp_query_emb
        graphFeature_map = torch.cat([support_graphFeature, query_graphFeature], 0)
        reconGraphFeature = self.graphFeatureRecon(graphFeature_map)

        return logits, pred, reconGraphFeature


