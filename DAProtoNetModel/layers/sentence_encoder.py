import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))

from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
from DAProtoNetModel.layers.network.CNNEncoder import cnnEncoder

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                word_embedding_dim, pos_embedding_dim)
        self.in_cnn = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)  # 编码领域不变特征
        self.sp_cnn = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)  # 编码领域特定特征
        self.drop = nn.Dropout(0.2)
        self.word2id = word2id
        self.in_encoder = self._in_encoder
        self.sp_encoder = self._sp_encoder
    def forward(self, inputs):

        in_x = self._in_encoder(inputs["word"])
        sp_x = self._sp_encoder(inputs["word"])
        return in_x, sp_x

    def _in_encoder(self, inputs):
        x = self.embedding(inputs)
        x = self.drop(x)
        in_x = self.in_cnn(x)
        return in_x

    def _sp_encoder(self, inputs):
        x = self.embedding(inputs)
        x = self.drop(x)
        sp_x = self.sp_cnn(x)
        return sp_x

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
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.in_bert = BertModel.from_pretrained(pretrain_path)
        self.sp_bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.in_encoder = self._in_encoder
        self.sp_encoder = self._sp_encoder
        self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.drop = nn.Dropout(0.2)

    def forward(self, inputs):
        # 方式一
        # word_embed, _ = self.bert(inputs['word'])
        # word_embed = self.drop(word_embed)
        # in_x = self.in_cnn_encoder(word_embed)
        # sp_x = self.sp_cnn_encoder(word_embed)
        # 方式二
        in_x = self._in_encoder(inputs)# 是因为后面需要多态调用才增加的封装
        sp_x = self._sp_encoder(inputs)
        # in_x = self.drop(in_x)
        # sp_x = self.drop(sp_x)
        return in_x, sp_x

    def _in_encoder(self,inputs):
        _, in_x = self.in_bert(inputs['word'])
        return in_x

    def _sp_encoder(self, inputs):
        _, sp_x = self.sp_bert(inputs['word'])
        return sp_x

    def tokenize(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens




class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path,checkpoint_name, max_length, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)
        self.sp_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)
        self.in_roberta.train()
        self.sp_roberta.train()

        self.max_length = max_length
        # self.tokenizer = 直接只用in_roberta内置的方法 具体查看fairseq 的接口
        self.cat_entity_rep = cat_entity_rep

        self.in_encoder = self._in_encoder
        self.sp_encoder = self._sp_encoder
        # 后面的Bert特征怎么取到的问题
        # CNN的方案 如果不用那么就没必要增加GPU负担
        # self.in_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)
        # self.sp_cnn_encoder = network.CNNEncoder.cnnEncoder(max_length)

        # 调用分类器来获得语义信息方案
        # embedding_size = 768
        # self.in_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)
        # self.sp_roberta.register_classification_head('emb_sentence', num_classes=embedding_size)


    def forward(self, inputs):

        # ## 方式一 使用CNN
        # word_embed = self.in_roberta.extract_features(inputs['word'])#@jinhui 疑问：有没有办法可以限制它的更新？
        # # word_embed = self.drop(word_embed)
        # in_x = self.in_cnn_encoder(word_embed)
        # sp_x = self.sp_cnn_encoder(word_embed)
        #
        # return in_x, sp_x
        # # 方式二
        in_x = self._in_encoder(inputs)#B, S_N ,D
        sp_x = self._sp_encoder(inputs)
        return in_x, sp_x

        # # 方案三 调用分类器
        # in_x = self.in_roberta.predict('emb_sentence',inputs['word'])#B, D
        # sp_x = self.sp_roberta.predict('emb_sentence', inputs['word'])
        # return  in_x, sp_x # 4,768

    def _in_encoder(self,inputs):
        in_x = self.in_roberta.extract_features(inputs['word'])[:, 1, :]  # B, S_N ,D
        return in_x

    def _sp_encoder(self,inputs):
        sp_x = self.sp_roberta.extract_features(inputs['word'])[:, 1, :]#@jinhui 因为这个的缘故吗？ 重要的差异
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


class RobertaNewgraphSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path,checkpoint_name, max_length, graphFeature_size=100, hidden_size=768, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)

        self.max_length = max_length
        self.cat_entity_rep = cat_entity_rep

        self.in_encoder = self._in_encoder
        self.sp_encoder = self._sp_encoder
        # TODO jinhui
        # graph feature map
        # encoder
        self.graphFeaturetransfer = nn.Sequential(
            nn.Linear(graphFeature_size, hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size * 2, hidden_size))
        # self.g1_drop = nn.Dropout()

        # self.in_featuretransfer = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size * 2, hidden_size))



    def forward(self, inputs):
        in_x = self._in_encoder(inputs)#B, S_N ,D
        sp_x = self._sp_encoder(inputs)

        return in_x, sp_x

    def _in_encoder(self, inputs):
        in_x = self.in_roberta.extract_features(inputs['word'])[:,1,:]  # B, S_N ,D

        return in_x
    def _sp_encoder(self,inputs):
        sp_x = self.graphFeaturetransfer(self.getGraphFeature(inputs))
        return sp_x
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

    def getGraphFeature(self, inputs):
        # 直接返回的方案
        return inputs["graphFeature"]


class RobertaNewgraphSentenceEncoder_old(nn.Module):
    def __init__(self, pretrain_path,checkpoint_name, max_length, graphFeature_size=100, hidden_size=768, cat_entity_rep=False):
        nn.Module.__init__(self)
        # 也可以考虑直接取Bert CSL 和SLP的方案
        from fairseq.models.roberta import RobertaModel
        self.in_roberta = RobertaModel.from_pretrained(pretrain_path, checkpoint_file=checkpoint_name)# 差异性需要有，但是功能需要用接口封装（相同的函数）

        self.max_length = max_length
        self.cat_entity_rep = cat_entity_rep

        # TODO jinhui
        # graph feature map
        # encoder
        self.graphFeaturetransfer = nn.Sequential(
            nn.Linear(graphFeature_size, hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size * 2, hidden_size))
        self.g1_drop = nn.Dropout()

    def forward(self, inputs):
        in_x = self.in_roberta.extract_features(inputs['word'])#B, S_N ,D
        sp_x = self.graphFeaturetransfer(self.getGraphFeature(inputs))

        return in_x[:,1,:], sp_x



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

    def getGraphFeature(self, inputs):
        # 直接返回的方案
        return inputs["graphFeature"]