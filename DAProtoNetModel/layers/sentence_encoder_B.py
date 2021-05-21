import sys,os,torch
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(os.path.abspath(__file__)), '..')))
import torch.nn as nn


from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
from DAProtoNetModel.layers.network.CNNEncoder import cnnEncoder


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path,checkpoint_name, max_length, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)

        self.max_length = max_length
        # self.tokenizer = 直接只用in_roberta内置的方法 具体查看fairseq 的接口
        self.cat_entity_rep = cat_entity_rep

        self.in_encoder = self._in_encoder

        # 后面的Bert特征怎么取到的问题
        # CNN的方案 如果不用那么就没必要增加GPU负担
        # self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)

        # 调用分类器来获得语义信息方案
        # embedding_size = 768
        # self.in_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)
        # self.sp_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)


    def forward(self, inputs):


        in_x = self._in_encoder(inputs)#B, S_N ,D

        return in_x


    def _in_encoder(self,inputs):
        in_x = self.in_roberta.extract_features(inputs['word'])[:, 1, :]  # B, S_N ,D
        return in_x

    def _sp_encoder(self,inputs):
        sp_x = self.sp_roberta.extract_features(inputs['word'])[:, 1, :]#@jinhui
        return  sp_x


    def tokenize(self, raw_tokens):
        # token -> index #查看到raw_tokens本身有CLS

        tokens = 'CLS'
        for token in raw_tokens:
            tokens += " " + token
        indexed_tokens = self.in_roberta.encode(tokens).tolist()


        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(2)#该用什么填充呢？
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.in_bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.in_encoder = self._in_encoder
        self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)

        # self.drop = nn.Dropout(0.2)

    def forward(self, inputs):

        in_x = self._in_encoder(inputs)  # 是因为后面需要多态调用才增加的封装

        return in_x

    def _in_encoder(self, inputs):
        _, in_x = self.in_bert(input_ids=inputs['word'],attention_mask=inputs['attention_mask'])#2jinhui 改 加入attentionmask
        return in_x

    def tokenize(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        attention_mask = [1] * len(indexed_tokens)
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
            attention_mask.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        return indexed_tokens, attention_mask


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)
        self.in_cnn = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim,
                                              hidden_size)  # 编码领域不变特征

        self.drop = nn.Dropout(0.2)
        self.word2id = word2id
        self.in_encoder = self._in_encoder
        # self.sp_encoder = self._sp_encoder

    def forward(self, inputs):

        in_x = self._in_encoder(inputs)
        return in_x

    def _in_encoder(self, inputs):
        x = self.embedding(inputs)
        x = self.drop(x)
        in_x = self.in_cnn(x)
        return in_x

    def tokenize(self, raw_tokens):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        attention_mask = [1 for i in range(len(indexed_tokens))]
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
            attention_mask.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        return indexed_tokens, attention_mask


# import torch
# from torch import nn
class LSTMSentenceEncoder(nn.Module):
    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        super(LSTMSentenceEncoder, self).__init__()
        # 製作 embedding layer
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.drop = nn.Dropout(0.2)
        self.word2id = word2id
        self.lstm = nn.LSTM(word_embedding_dim, hidden_size, num_layers=1, batch_first=True)
        # old
        # self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        # self.embedding.weight.requires_grad = False if fix_embedding else True
        # self.embedding_dim = embedding.size(1)
        # self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        # self.dropout = dropout
        # self.lstm = nn.LSTM(word_embedding_dim, hidden_size, num_layers=4, batch_first=True)
        # self.classifier = nn.Sequential( nn.Dropout(0.2),
        #                                  nn.Linear(hidden_size, 2),)

    def forward(self, inputs):
        inputs = self.embedding(inputs)# nan, nan?
        inputs = inputs.transpose(0,1)#input of shape (seq_len, batch, input_size)
        self.drop(inputs)
        x, _ = self.lstm(inputs, None)# output of shape (seq_len, batch, input_size)

        x = torch.mean(x, 0)
        # x 的 dimension (seq_len,batch, hidden_size)
        return x

    def tokenize(self, raw_tokens):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        attention_mask = [1 for i in range(len(indexed_tokens))]
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
            attention_mask.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        return indexed_tokens, attention_mask
