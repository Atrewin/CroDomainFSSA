
import torch
from torch import optim
import numpy as np
import json
import argparse
import os
from DAProtoNetModel.layers.data_loader import get_loader, get_loader_unsupervised
from DAProtoNetModel.layers.framework import FewShotREFramework
from DAProtoNetModel.layers.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
from DAProtoNetModel.layers.DAProtoNet import DAProtoNet
from DAProtoNetModel.layers.d import Discriminator
from DAProtoNetModel.layers.sen_d import Sen_Discriminator

def main():

    parser = argparse.ArgumentParser()

    # data url parameters
    parser.add_argument('--train', default='book_reviews', help='train file')
    parser.add_argument('--val', default='dvd_reviews', help='val file')
    parser.add_argument('--test', default='dvd_reviews', help='test file')
    parser.add_argument('--adv', default=None, help="adv unlabeded reviews files")

    # model training parameters
    parser.add_argument('--trainN', default=2, type=int, help='N in train')  # 固定trainN=2
    parser.add_argument('--N', default=2, type=int, help='N way')  # 固定N=2
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int, help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--pretrain_step', default=500, type=int, help="pretrain_step")
    parser.add_argument('--train_iter', default=20000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int, help='val after training how many iters')
    parser.add_argument('--encoder', default='cnn', help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=200, type=int, help='max length')  # 数据集的平均长度
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd', help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int, help='hidden size')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--fp16', action='store_true', help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true', help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true', help='do not use dropout after BERT (still has dropout in BERT).')
    
    # experiment
    parser.add_argument('--mask_entity', action='store_true', help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = "DAProtoNet"
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length)  # 编码领域不变特征
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt
        filepath, tempfilename = os.path.split(pretrain_ckpt)
        sentence_encoder =RobertaSentenceEncoder(filepath,tempfilename, max_length, cat_entity_rep=opt.cat_entity_rep)
    else:
        raise NotImplementedError

    train_data_loader = get_loader(opt.train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    test_data_loader = get_loader(opt.test, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    if opt.adv:
        adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    # 领域二分类器
    d = Discriminator(opt.hidden_size)
    sen_D = Sen_Discriminator(opt.hidden_size)# 是这里开始变慢了？
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader=adv_data_loader, adv=opt.adv, d=d, sen_d=sen_D)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])

    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    #  构造模型   @jinhui 将整个模型框架传入的设计会更加优雅
    model = DAProtoNet(sentence_encoder, opt.hidden_size, dot=opt.dot)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()# 疑问 这一句有成功tocuda吗？

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1
        # @jinhui 这里的传参模式不符合面向对象程序设计思想, 建议作为属性成员传入framework
        framework.train(model, prefix, batch_size, trainN, N, K, Q, pretrain_step=opt.pretrain_step,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16,
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt)
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
