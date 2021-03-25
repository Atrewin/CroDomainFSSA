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

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None,
                 sen_d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.sen_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
            self.sen_d = sen_d
            self.sen_d.cuda()

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
              use_sgd_for_bert=False):
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
        print("Start training...")

        # for name, param in model.named_parameters():
        #     print(name)
        # exit()

        # TODO optimizer_encoder
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())  # 变量格式? tuple(name: str,param: contain)

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # @jinhui 疑问点: 这里为什么要这样设置leanable parameters weight_decay
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            # 优化器和参数绑定
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)

            # @jinhui 疑惑:这里不会导致parameters_to_optimize和多个optimizer绑定吗?
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)# 应该只限制到

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                        num_training_steps=train_iter)
        else:
            optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)

            if self.adv:
                not_upgrade_params = ["fc.weight", "fc.bias"]  # 怎么知道这两参数不用更新的
                for name, param in model.named_parameters():  # 指针对象?
                    if name in not_upgrade_params:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)
            optimizer_sen_dis = pytorch_optim(self.sen_d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
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

        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):

            support, query, label = next(self.train_data_loader)
            if torch.cuda.is_available():  # @jinhui 疑惑 为什么要分开to cuda
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()

            # 调用模型 # Prototypical 的输出 (B, total_Q, N + 1)
            logits, pred = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)
            # 重构graphFeature
            support_graphFeature, query_graphFeature = support["graphFeature"], query["graphFeature"]
            graphFeature = torch.cat([support_graphFeature, query_graphFeature], 0)

            graphFeatureRcon = model.graphFeatureRecon(graphFeature)
            loss_recon = model.loss_recon(graphFeatureRcon, graphFeature) * 0.5 #可能影响太大了

            loss = model.loss(logits, label) / float(grad_iter)
            loss = loss+ loss_recon
            right = model.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:  # @jinhui 貌似这个就是用来累计梯度的
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # @jinhui 疑惑: 感觉上面流程已经结束, 为什么还要对抗?
            if self.adv and it > pretrain_step:  # 对抗策略
                support_adv = next(self.adv_data_loader)  # 拿
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                # 这样会使得你的sentence encoder之时encode出领域不变特征。而忽略了领域特定特征 #@jinhui 书写建议是写入到DaProtoNet内部
                features_ori, _ = model.sentence_encoder(support)  # (B*2*K, hidden_size)
                features_adv, _ = model.sentence_encoder(support_adv)
                support_total = features_ori.size(0)
                features = torch.cat([features_ori, features_adv], 0)  # 上下拼接
                total = features.size(0)  #
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                                        torch.ones((total // 2)).long().cuda()], 0)
                # @jinhui 如果support_total 为基数会发生什么? 虽然这里way = 2 是不会发生的
                sentiment_labels = torch.cat([torch.zeros((support_total // 2)).long().cuda(),
                                              torch.ones((support_total // 2)).long().cuda()], 0)

                dis_logits = self.d(features)
                sen_logits = self.sen_d(features_ori)# 这里貌似有点问题，不应该是这里可以分辨情绪
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                # print(loss_dis) # tensor(0.7033, device='cuda:0', grad_fn=<NllLossBackward>)
                sen_loss_dis = self.sen_cost(sen_logits, sentiment_labels)
                _, pred = dis_logits.max(-1)
                _, sen_pred = sen_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                sen_right_dis = float((sen_pred == sentiment_labels).long().sum()) / float(support_total)

                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()  # 改变了模型参数, 导致后面的loss_encoder.backward 出现了 inplace operater
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()
                optimizer_sen_dis.zero_grad()
                optimizer.zero_grad()

                sen_loss_dis.backward(retain_graph=True)
                optimizer_sen_dis.step()

                optimizer_sen_dis.zero_grad()
                optimizer_dis.zero_grad()
                # optimizer_encoder.zero_grad()#@jinhui 改 应该同步优化  所以它的loss先不清零
                optimizer.zero_grad()

                # 不同label [1, 0], 目的是为了得到的表示feature无法区分domain, 也因此right_dis的准确率处于50%上下浮动
                dis_logits = self.d(features)  # @jinhui features要是保持之前的in_spect的梯度跟踪吗？
                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)  # # 这里已经不能够使用了, 原因是计算dis_logits的模型参数已经变化
                # print(loss_encoder)# tensor(0.6985, device='cuda:0', grad_fn=<NllLossBackward>)
                loss_encoder.backward(retain_graph=True)  # torch 上面的loss_dis可以通过
                # 可能的原因是，forward 和 backward前后有torch被inplace operarion 了
                # .data 与 .detach()中可能在1.7版本中梯度的检查是不一样的
                optimizer_encoder.step()  # 现在这里还包含最开始的loss 梯度在, 不知道要不要改，预计sent_acc会提高到80%以上
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()
                optimizer_sen_dis.zero_grad()
                optimizer.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis
                iter_sen_right_dis += sen_right_dis

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            if self.adv:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}, sen_acc: {5:2.6f}'
                    .format(it + 1, iter_loss / iter_sample,
                            100 * iter_right / iter_sample,
                            iter_loss_dis / iter_sample,
                            100 * iter_right_dis / iter_sample,
                            100 * iter_sen_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                               100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter,
                                na_rate=na_rate)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sen_right_dis = 0.
                iter_sample = 0.
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
             model,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
             ckpt=None):
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
        print("")

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
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
            for it in range(eval_iter):
                support, query, label = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write(
                    '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
