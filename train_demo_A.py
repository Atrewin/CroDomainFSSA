from torch import optim
from utils.model_utils import *
import datetime, torch
import argparse
from DAProtoNetModel.layers.data_loader_A import get_loader, get_loader_unsupervised,get_loader_val
from DAProtoNetModel.layers.framework_A import FewShotREFramework
from DAProtoNetModel.layers.sentence_encoder_A import *  #改
from DAProtoNetModel.layers.DAPostBertNet_A import DAPostBertNet
from DAProtoNetModel.layers.d import Discriminator
from DAProtoNetModel.layers.sen_d import Sen_Discriminator, Sen_Discriminator_sp
import traceback
from utils.logger import *

def main():
    parser = argparse.ArgumentParser()

    # data url parameters
    parser.add_argument('--train', default='books/book_reviews', help='train file')
    parser.add_argument('--val', default='dvd/dvd_reviews', help='val file')
    parser.add_argument('--test', default='dvd/dvd_reviews', help='test file')
    parser.add_argument('--adv', default=None, help="adv unlabeded reviews files")
    # 几点
    # model training parameters
    parser.add_argument('--trainN', default=2, type=int, help='N in train')  # 固定trainN=2
    parser.add_argument('--N', default=2, type=int, help='N way')  # 固定N=2
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int, help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int, help='val after training how many iters')
    parser.add_argument('--encoder', default='roberta', help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=200, type=int, help='max length')  # 数据集的平均长度
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw', help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int, help='hidden size')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--fp16', action='store_true', help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true',
                        help='do not use dropout after BERT (still has dropout in BERT).')

    # experiment
    parser.add_argument('--mask_entity', action='store_true', help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')

    # 切分阶段设计
    parser.add_argument('--start_train_prototypical', default=-1, type=int, help='iter to start train prototypical')
    parser.add_argument('--start_train_adv', default=500, type=int, help="iter to start add adv")
    parser.add_argument('--start_train_dis', default=-1, type=int, help='iter to start train discriminator.')

    # log模块设计
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    parser.add_argument("--notes", type=str, default='', help='Root directory for all logging.')
    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = "DAProtoNet"
    encoder_name = opt.encoder
    max_length = opt.max_length

    # train_log setting
    LOG_PATH = opt.log_root
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])

    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, prefix + opt.notes + nowTime + ".txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    logger.info("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    logger.info("log_path: {}".format(log_path))
    logger.info("model: {}".format(model_name))
    logger.info("encoder: {}".format(encoder_name))
    logger.info("max_length: {}".format(max_length))
    #  构造模型   @jinhui 将整个模型框架传入的设计会更加优雅

    if encoder_name == 'roberta_A':#改
        pretrain_ckpt = opt.pretrain_ckpt
        max_length = opt.max_length
        filepath, tempfilename = os.path.split(pretrain_ckpt)
        sentence_encoder = RobertaSentenceEncoder(filepath, tempfilename, max_length, cat_entity_rep=opt.cat_entity_rep)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        max_length = opt.max_length
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)


    model = DAPostBertNet(sentence_encoder, opt.hidden_size, dot=opt.dot)#改

    train_data_loader = get_loader(opt.train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=opt.na_rate,
                                   batch_size=batch_size)
    val_data_loader = get_loader_val(opt.val, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, data_mode="val")
    test_data_loader = get_loader_val(opt.test, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, data_mode="test")

    adv_data_loader = None
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
    sen_D = Sen_Discriminator(opt.hidden_size)  # 是这里开始变慢了？
    sen_sp_D = Sen_Discriminator_sp(opt.hidden_size)
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader,
                                   adv_data_loader=adv_data_loader, adv=opt.adv, d=d, sen_d=sen_D, sen_sp_D=sen_sp_D)

    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])

    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()  # 疑问 这一句有成功tocuda吗？
    try:

        if not opt.only_test:
            if encoder_name in ['bert', 'roberta', "roberta_newGraph", 'roberta_A']:
                bert_optim = True
            else:
                bert_optim = False

            if opt.lr == -1:
                if bert_optim:
                    opt.lr = 2e-5
                else:
                    opt.lr = 1e-1
            # @jinhui 这里的传参模式不符合面向对象程序设计思想, 建议作为属性成员传入framework
            framework.train(model, prefix, batch_size, trainN, N, K, Q,
                            pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                            na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16,
                            train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim,
                            learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, opt=opt)
        else:
            ckpt = opt.load_ckpt
            if ckpt is None:
                logger.info("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
                ckpt = 'none'
    except RuntimeError:  # 提前中断，执行最终测试

        logger.debug(traceback.print_exc())
        acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt)
        logger.info("RESULT: %.2f" % (acc * 100))

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt)
    logger.info("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
