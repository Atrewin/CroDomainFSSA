import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
import traceback
from utils.logger import *
from utils.view import *
from utils.json_util import *
def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder, hidden_size):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.hidden_size = hidden_size
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
        # return self.cost(F.softmax(logits.view(-1, N),dim=1), label.view(-1))#jinhui 改0507

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))#0.2500,


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None,
                 sen_d=None, sen_sp_D=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        # 准备训练数据
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv

        #准备训练模型
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.sen_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()# 不应该在这里tocuda
            self.sen_d = sen_d
            self.sen_d.cuda()
            self.sen_sp_d = sen_sp_D
            self.sen_sp_d.cuda()

        # 准备训练优化器（放到了train阶段）

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              pretrain_step=500,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False,
              opt=None):
        '''
        model: a FewShotREModel instance  #@jinhui 这里只是sentence_encoder
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        logger.info("Start training...")

        # for name, param in model.named_parameters():
        #     print(name)
        # exit()

        # TODO optimizer_encoder
        if bert_optim:
            logger.info('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())  # 变量格式? tuple(name: str,param: contain)

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # @jinhui 疑问点: 这里为什么要这样设置leanable parameters weight_decay
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            # @改 除了bert 其他模块应该有更高的学习率 @改 0422
            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # low_lr = ["sentence_encoder"]
            # # @jinhui 疑问点: 这里为什么要这样设置leanable parameters weight_decay
            # parameters_to_optimize = [
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if (not any(nd in n for nd in no_decay)) and (any(low in n for low in low_lr))], 'weight_decay': 0.01},
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if any(nd in n for nd in no_decay) and (any(low in n for low in low_lr))], 'weight_decay': 0.0},
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if (not any(nd in n for nd in no_decay)) and ( not any(low in n for low in low_lr))],
            #      'weight_decay': 0.01, 'lr': 1e-1},# bert之外的人应该有更多的学习率
            #     {'params': [p for n, p in parameters_to_optimize
            #                 if any(nd in n for nd in no_decay) and (not any(low in n for low in low_lr))],
            #      'weight_decay': 0.0, 'lr': 1e-1}
            # ]

            # 优化器和参数绑定
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                        num_training_steps=train_iter)

            # @jinhui 疑惑:这里不会导致parameters_to_optimize和多个optimizer绑定吗?
            # if self.adv:
            #     optimizer_encoder = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)# 应该只限制到

        else:
            optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

            # if self.adv:
            #     not_upgrade_params = ["fc.weight", "fc.bias"]  # 怎么知道这两参数不用更新的
            #     for name, param in model.named_parameters():  # 指针对象?
            #         if name in not_upgrade_params:
            #             param.requires_grad = False
            #         else:
            #             param.requires_grad = True
            #     optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)


        if self.adv:
            # optimizer_dis = AdamW(self.d.parameters(), lr=adv_dis_lr, correct_bias=False)
            optimizer_dis = AdamW(self.d.parameters(), lr=adv_dis_lr)
            optimizer_sen_dis = AdamW(self.sen_d.parameters(), lr=adv_dis_lr)
            optimizer_sen_dis_sp = AdamW(self.sen_sp_d.parameters(), lr=adv_dis_lr, correct_bias=False)
        if load_ckpt:
            load_ckpt = save_ckpt
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    logger.info('ignore {}'.format(name))
                    continue
                logger.info('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sen_right_dis = 0.0
        iter_sen_right_sp = 0.0

        iter_sample = 0.0
        iter_proto = 1
        iter_dis = 1

        right_dis = 1
        for it in range(start_iter, start_iter + train_iter):

            support, query, label, support_label = next(self.train_data_loader)
            if torch.cuda.is_available():  # @jinhui 疑惑 为什么要分开to cuda 'dict' object has no attribute 'cuda'
                label = label.cuda()
                support_label = support_label.cuda()
                for k in support:
                    support[k] = support[k].cuda()# 这里直接cuda()会有问题？
                for k in query:
                    query[k] = query[k].cuda()

                # support = support.cuda()
                # query = query.cuda()
            # 重构graphFeature
            support_graphFeature, query_graphFeature = support["graphFeature"], query["graphFeature"]
            graphFeature = torch.cat([support_graphFeature, query_graphFeature], 0)

            if it > opt.start_train_prototypical:

                # 调用模型 # Prototypical 的输出 (B, total_Q, N + 1)
                if opt.encoder in ["roberta_newGraph","bert_newGraph", "graph"]:

                    if opt.ignore_graph_feature:
                        logits, pred = model.forwardWithIgnoreGraph(support, query, N_for_train, K,
                                             Q * N_for_train + na_rate * Q)
                        loss = model.loss(logits, label) / float(grad_iter)
                        pass
                    elif opt.ignore_bert_feature:# 为了减少内存的使用，特意重写了一个sentence encoder（graph）
                        logits, pred, graphFeatureRcon = model.forwardWithIgnoreBert(support, query, N_for_train, K,
                                                                    Q * N_for_train + na_rate * Q)
                        loss_recon = model.loss_recon(graphFeatureRcon, graphFeature) * 0.05  # 可能影响太大了
                        loss = model.loss(logits, label)
                        loss = (loss + loss_recon) / float(grad_iter)
                    else:
                        logits, pred, graphFeatureRcon = model.forwardWithRecon(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)

                        loss_recon = model.loss_recon(graphFeatureRcon, graphFeature) * 0.05 #可能影响太大了
                        loss = model.loss(logits, label)
                        loss = (loss + loss_recon) / float(grad_iter)
                else:
                    logits, pred= model(support, query, N_for_train, K,
                                                                            Q * N_for_train + na_rate * Q)
                    loss = model.loss(logits, label) / float(grad_iter)

                right = model.accuracy(pred, label)
                # loss = loss*6#@jinhui 统一调整学习率的做法
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
                else:
                    # sum_loss = loss
                    loss.backward(retain_graph=True)

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                if it % grad_iter == 0:  # @jinhui 貌似这个就是用来累计梯度的
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # torch.cuda.empty_cache()

                iter_loss += self.item(loss.data)
                iter_right += self.item(right.data)
                iter_proto += 1
            # @jinhui 疑惑: 感觉上面流程已经结束, 为什么还要对抗?


            # TODO 对抗训练
            if self.adv and it > opt.start_train_adv:  # < 优先对抗策略 >后对抗策略

                if it == opt.start_train_adv + 1:
                    # logger.info('\n')
                    sen_d_alpha = 1
                    sen_sp_d_alpha = 1

                # 动态调整对抗的梯度差异
                if right_dis >= 0.5:# 只有发现差异才需要拉近 @jinhui
                    len_dataloader = 1000
                    n_epochs = 10000
                    p = float(it + it * len_dataloader) / n_epochs / len_dataloader

                    alpha = 2. / (1. + np.exp(-10 * p)) - 1  # 必须大于0，不然梯度就没有反转了
                else:
                    alpha = 0.0000000000001



                # optimizer_encoder.zero_grad()
                optimizer.zero_grad()
                support_adv = next(self.adv_data_loader)  # 拿
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()
                #@jinhui 问题：[:,1,:] 1是否会出问题
                # 这样会使得你的sentence encoder之时encode出领域不变特征。而忽略了领域特定特征 #@jinhui 书写建议是写入到DaProtoNet内部
                support_features = model.sentence_encoder(support)  # (B*2*K, hidden_size)# 这里选这从新forward（没必要学KIONG那样增加了强耦合，只为了减少一次forward）
                features_ori = support_features[0]
                sp_support_features = support_features[1]
                features_adv = model.sentence_encoder.module.in_encoder(support_adv)# @jinhui
                support_total = features_ori.size(0)
                features = torch.cat([features_ori, features_adv], 0)  # 上下拼接
                total = features.size(0)  #
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                                        torch.ones((total // 2)).long().cuda()], 0)
                # @jinhui 如果support_total 为基数会发生什么? 虽然这里way = 2 是不会发生的
                # sentiment_labels = torch.cat([torch.zeros((support_total // 2)).long().cuda(),
                #                               torch.ones((support_total // 2)).long().cuda()], 0)# 这里的acc一直很差, 是这里的label没有和真实的label对齐
                sentiment_labels = support_label
                dis_logits = self.d(features, alpha=alpha)  # d内部含有梯度反转层， 采用动态设计
                sen_logits = self.sen_d(features_ori, sen_d_alpha)# 这里貌似有点问题，不应该是这里可以分辨情绪
                sen_logits_sp = self.sen_sp_d(sp_support_features, sen_sp_d_alpha)


                # logsoftmax_func = nn.LogSoftmax(dim=1)
                # dis_logits = logsoftmax_func(dis_logits)
                # sen_logits = logsoftmax_func(sen_logits)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                sen_loss_dis = self.sen_cost(sen_logits, sentiment_labels)# 怎么这么大？没有softmax?
                sen_loss_sp = self.sen_cost(sen_logits_sp, sentiment_labels)
                if it > opt.start_train_dis :

                    if it == opt.start_train_dis + 1:#这时候sen loss只更新sen_d
                        sen_d_alpha = 0
                        sen_sp_d_alpha = 0

                    sum_loss = (sen_loss_dis + loss_dis + sen_loss_sp) / 3 #20/21# 这里是没办法调节两者比例的，因为优化器是AdamW
                else:
                    sum_loss = (sen_loss_dis + loss_dis + sen_loss_sp) / 3
                sum_loss.backward(retain_graph=True)# retain_graph=True
                if it % grad_iter == 0:  # @jinhui 貌似这个就是用来累计梯度的
                    optimizer_dis.step()
                    optimizer_sen_dis.step()
                    optimizer_sen_dis_sp.step()
                    optimizer.step()

                    optimizer.zero_grad()
                    optimizer_dis.zero_grad()
                    optimizer_sen_dis.zero_grad()
                    optimizer_sen_dis_sp.zero_grad()
                    # optimizer_encoder.zero_grad()# s是否想要双优化器想要讨论
                    # torch.cuda.empty_cache()



                _, pred = dis_logits.max(-1)
                _, sen_pred = sen_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                sen_right_dis = float((sen_pred == sentiment_labels).long().sum()) / float(support_total)

                # #TODO sp_encoder sentiment
                # sp_features = model.sentence_encoder.module.sp_encoder(query)
                # query_total = sp_features.size(0)
                # sen_logits_sp = self.sen_d(sp_features)
                _, sen_pred_sp = sen_logits_sp.max(-1)
                sen_right_sp = float((sen_pred_sp == sentiment_labels).long().sum()) / float(support_total)


                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis
                iter_sen_right_dis += sen_right_dis
                iter_sen_right_sp += sen_right_sp
                iter_dis += 1

            iter_sample += 1

            if self.adv:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}, sen_in_acc: {5:2.6f}, sen_sp_acc: {5:2.6f}'
                    .format(it + 1, iter_loss / iter_proto,
                            100 * iter_right / iter_proto,
                            iter_loss_dis / iter_dis,
                            100 * iter_right_dis / iter_dis,
                            100 * iter_sen_right_dis / iter_dis,
                            100 * iter_sen_right_sp / iter_dis,) + '\r')
            else:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_proto,
                                                                               100 * iter_right / iter_proto) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                #log
                if self.adv:
                    logger.info(
                        'step: {0:4} | loss: {1:2.6f}, accuracy_: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}, sen_in_acc: {5:2.6f}, sen_sp_acc: {5:2.6f}'
                        .format(it + 1, iter_loss / iter_proto,
                                100 * iter_right / iter_proto,
                                iter_loss_dis / iter_dis,
                                100 * iter_right_dis / iter_dis,
                                100 * iter_sen_right_dis / iter_dis,
                                100 * iter_sen_right_sp / iter_dis, ) + '\r')
                else:
                    logger.info(
                        'step: {0:4} | loss: {1:2.6f}, accuracy_: {2:3.2f}%'.format(it + 1, iter_loss / iter_proto,
                                                                                   100 * iter_right / iter_proto) + '\r')

                acc = self.eval(model, B, N_for_eval, K, Q, val_iter,
                                na_rate=na_rate, opt=opt)
                optimizer.zero_grad()  # 防止eval()的时候发生波动
                model.train()
                if acc > best_acc:
                    logger.info('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sen_right_dis = 0.
                iter_sen_right_sp = 0.
                iter_sample = 0.
                iter_proto = 1
                iter_dis = 1
                logger.info("\n")
        logger.info("\n####################\n")
        logger.info("Finish training " + model_name)

    def eval(self,
               model,
               B, N, K, Q,
               eval_iter,
               na_rate=0,
               ckpt=None, opt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        # print("")
        model.eval()  # @jinhui 难道这里model.eval() ：不启用 BatchNormalization 和 Dropout 但是如果使用的是self.training = mode 估计只是作用到了model.forward()函数上了
        if ckpt is None:
            logger.info("Use val dataset")
            eval_dataset = self.val_data_loader  # @jinhui
        else:
            logger.info("Use test dataset")
            if ckpt != 'none':

                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            eval_dataset.dict_val = {}
            for it in range(eval_iter):
                # jinhui check
                try:
                    support, query, label, _ = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        label = label.cuda()
                    # logits, pred = model(support, query, N, K, Q * N + Q * na_rate)#@jinhui bug 5.7前着地方是这样子的

                    if opt.encoder in ["roberta_newGraph", "bert_newGraph", "graph"]:
                        # 自定义的forward脱离了model.eval()的监督，导致梯度波动
                        if opt.ignore_graph_feature:
                            logits, pred = model.forwardWithIgnoreGraph(support, query, N, K,
                                                                        Q * N + na_rate * Q)

                        elif opt.ignore_bert_feature:  # 为了减少内存的使用，特意重写了一个sentence encoder（graph）
                            logits, pred, graphFeatureRcon = model.forwardWithIgnoreBert(support, query, N, K,
                                                                                         Q * N + na_rate * Q)
                        else:
                            logits, pred, graphFeatureRcon = model.forwardWithRecon(support, query, N, K,
                                                                                    Q * N + na_rate * Q)
                    else:
                        logits, pred = model(support, query, N, K, Q * N + na_rate * Q)

                    right = model.accuracy(pred, label)
                    iter_right += self.item(right.data)  # 除非这里包含这保留三位小数的操作
                    iter_sample += 1

                    sys.stdout.write(
                        '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1,
                                                                          100 * iter_right / iter_sample) + '\r')
                    sys.stdout.flush()
                except RuntimeError:
                    logger.info(it)
                    continue
            logger.info(
                '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')

        return iter_right / iter_sample


    def visual(self,
             model,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
             ckpt=None, opt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        # print("")
        model.eval()#@jinhui 难道这里model.eval() ：不启用 BatchNormalization 和 Dropout 但是如果使用的是self.training = mode 估计只是作用到了model.forward()函数上了
        if ckpt is None:
            logger.info("Use val dataset")
            eval_dataset = self.val_data_loader#@jinhui
        else:
            logger.info("Use test dataset")
            if ckpt != 'none':
                if "checkpoint/" not in ckpt:
                    ckpt1 = "checkpoint/" + ckpt
                state_dict = self.__load_model__(ckpt1)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            storage_list = {
                "S_domain": {
                    "positive": [],
                    "negative": []
                },
                "T_domain": {
                    "positive": [],
                    "negative": []
                }
            }
            tips = True
            for it in range(eval_iter):
                  # jinhui check
                try:
                    support, query, label, _ = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        label = label.cuda()
                    # logits, pred = model(support, query, N, K, Q * N + Q * na_rate)#@jinhui bug 5.7前着地方是这样子的

                    if opt.encoder in ["roberta_newGraph", "bert_newGraph", "graph"]:
                        #自定义的forward脱离了model.eval()的监督，导致梯度波动
                        if opt.ignore_graph_feature:
                            total_Q = Q * N + na_rate * Q
                            in_support_emb, sp_support_emb = model.sentence_encoder(
                                support)  # (B * N * K, D), where D is the hidden size
                            in_query_emb, sp_query_emb = model.sentence_encoder(query)  # (B * total_Q, D)

                            hidden_size = in_support_emb.size(-1)

                            support_emb = in_support_emb
                            query_emb = in_query_emb

                            support = model.drop(support_emb)
                            query = model.drop(query_emb)
                            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
                            query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

                            # Prototypical Networks
                            # Ignore NA policy
                            support = torch.mean(support, 2)  # Calculate prototype for each class
                            logits = model.__batch_dist__(support, query)  # (B, total_Q, N)
                            minn, _ = logits.min(-1)
                            logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
                            _, pred = torch.max(logits.view(-1, N + 1), 1)  # N + 1会有问题吗？

                            # TODO visualization

                            # to list

                            if it < 500:
                                support_emb = support_emb.reshape(N, K, hidden_size)
                                points = support_emb.tolist()
                                storage_list["T_domain"]["positive"].extend(points[0])
                                storage_list["T_domain"]["negative"].extend(points[1])
                            else:
                                if tips:
                                    eval_dataset = self.test_data_loader
                                    tips = False
                                    continue

                                support_emb = support_emb.reshape(N, K, hidden_size)
                                points = support_emb.tolist()
                                storage_list["S_domain"]["positive"].extend(points[0])
                                storage_list["S_domain"]["negative"].extend(points[1])

                                if it > 998:
                                    keep = opt.encoder + opt.notes + ".json"
                                    dump(storage_list, keep)
                                    break

                        elif opt.ignore_bert_feature:  # 为了减少内存的使用，特意重写了一个sentence encoder（graph）
                            logits, pred, graphFeatureRcon = model.forwardWithIgnoreBert(support, query, N, K,
                                                                                         Q * N + na_rate * Q)
                        else:
                            total_Q = Q * N + na_rate * Q
                            in_support_emb, sp_support_emb = model.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
                            in_query_emb, sp_query_emb = model.sentence_encoder(query)  # (B * total_Q, D)
                            hidden_size = in_support_emb.size(-1)

                            support_emb = torch.cat([in_support_emb, sp_support_emb], axis=1)  # (B*N*K, 2D)
                            query_emb = torch.cat([in_query_emb, sp_query_emb], axis=1)  # (B*Q*N, 2D)
                            support_emb = model.fc(support_emb)
                            query_emb = model.fc(query_emb)

                            # support = model.drop(support_emb)
                            # query = model.drop(query_emb)
                            support = support_emb
                            query = query_emb
                            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
                            query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

                            # Prototypical Networks
                            # Ignore NA policy
                            Proto_support = torch.mean(support, 2)  # Calculate prototype for each class
                            logits = model.__batch_dist__(Proto_support, query)  # (B, total_Q, N)
                            minn, _ = logits.min(-1)
                            logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
                            _, pred = torch.max(logits.view(-1, N + 1), 1)  # N + 1会有问题吗？

                            #TODO visualization

                            #to list

                            if it < 500:
                                support_emb = support_emb.reshape(N, K, hidden_size)
                                points = support_emb.tolist()
                                storage_list["T_domain"]["positive"].extend(points[0])
                                storage_list["T_domain"]["negative"].extend(points[1])
                            else:
                                if tips:
                                    eval_dataset = self.test_data_loader
                                    tips = False
                                    continue

                                support_emb = support_emb.reshape(N, K, hidden_size)
                                points = support_emb.tolist()
                                storage_list["S_domain"]["positive"].extend(points[0])
                                storage_list["S_domain"]["negative"].extend(points[1])

                                if it > 998:
                                    keep = opt.encoder + opt.notes + ".json"
                                    dump(storage_list, keep)
                                    break




                            # 采用先保存的方案

                            # proto_points = proto_points.reshape(-1, 768).tolist()
                            # points_d, proto_points_d = view_on_two_dim(points, proto_points)
                            # mean = cal_mean(points_d, proto_points_d)
                            # R, mean_R = calc_radius(points_d, proto_points_d, scale=mean)#point_d

                    else:
                        # logits, pred = model(support, query, N, K, Q * N + na_rate * Q)
                        total_Q = Q * N + na_rate * Q
                        in_support_emb, sp_support_emb = model.sentence_encoder(
                            support)  # (B * N * K, D), where D is the hidden size
                        in_query_emb, sp_query_emb = model.sentence_encoder(query)  # (B * total_Q, D)
                        hidden_size = in_support_emb.size(-1)

                        """
                            学习领域不变特征  Learn Domain Invariant Feature
                        """
                        support_emb = torch.cat([in_support_emb, sp_support_emb], axis=1)  # (B*N*K, 2D)
                        query_emb = torch.cat([in_query_emb, sp_query_emb], axis=1)  # (B*Q*N, 2D)
                        support_emb = model.fc(support_emb)
                        query_emb = model.fc(query_emb)

                        support = model.drop(support_emb)
                        query = model.drop(query_emb)
                        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
                        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

                        # Prototypical Networks
                        # Ignore NA policy
                        Proto_support = torch.mean(support, 2)  # Calculate prototype for each class
                        logits = model.__batch_dist__(Proto_support, query)  # (B, total_Q, N)
                        minn, _ = logits.min(-1)
                        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
                        _, pred = torch.max(logits.view(-1, N + 1), 1)

                        # TODO visualization

                        # to list

                        if it < 500:
                            support_emb = support_emb.reshape(N, K, hidden_size)
                            points = support_emb.tolist()
                            storage_list["T_domain"]["positive"].extend(points[0])
                            storage_list["T_domain"]["negative"].extend(points[1])
                        else:
                            if tips:
                                eval_dataset = self.test_data_loader
                                tips = False
                                continue

                            support_emb = support_emb.reshape(N, K, hidden_size)
                            points = support_emb.tolist()
                            storage_list["S_domain"]["positive"].extend(points[0])
                            storage_list["S_domain"]["negative"].extend(points[1])

                            if it > 998:
                                keep = opt.encoder + opt.notes + ".json"
                                dump(storage_list, keep)
                                break

                    right = model.accuracy(pred, label)
                    iter_right += self.item(right.data)#除非这里包含这保留三位小数的操作
                    iter_sample += 1

                    sys.stdout.write(
                        '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                    sys.stdout.flush()
                except RuntimeError:
                    logger.info(it)
                    continue
            logger.info(
                        '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')

        return iter_right / iter_sample

