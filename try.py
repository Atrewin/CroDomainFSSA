#TODO Load RoBERTa from torch.hub (PyTorch >= 1.1):
import torch
import traceback

def exampel():
    # roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    # roberta.eval()  # disable dropout (or leave in train mode to finetune)
    import torch
    # Load the model in fairseq file #@jinhui 会重新修改开源文件
    from fairseq.models.roberta import RobertaModel
    roberta = RobertaModel.from_pretrained('./pre-train/roberta.large', checkpoint_file='model.pt')
    roberta.eval()  # disable dropout (or leave in train mode to finetune)

    # TODO 基础操作
    tokens = roberta.encode('Hello world!')
    assert tokens.tolist() == [0, 31414, 232, 328, 2]
    a = roberta.decode(tokens)  # 'Hello world!'

    # Extract the last layer's features
    last_layer_features = roberta.extract_features(tokens)
    assert last_layer_features.size() == torch.Size([1, 5, 1024])

    # Extract all layer's features (layer 0 is the embedding layer)
    all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
    assert len(all_layers) == 25
    assert torch.all(all_layers[-1] == last_layer_features)



    # TODO train example: Use RoBERTa for sentence-pair classification tasks:

    # Download RoBERTa already finetuned for MNLI
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.eval()  # disable dropout for evaluation

    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    roberta.predict('mnli', tokens).argmax()  # 0: contradiction#@jinhui 这个接口是可以修改为train的

    # Encode another pair of sentences
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    roberta.predict('mnli', tokens).argmax()  # 2: entailment

    # Register a new (randomly initialized) classification head:
    roberta.register_classification_head('new_task', num_classes=3)  # 加了一个分类器,叫new_task
    logprobs = roberta.predict('new_task',
                               tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)

    # Using the GPU:


    roberta2 = roberta.cuda()
    a = roberta2.predict('new_task', tokens)




    # TODO Batched prediction:
    import torch
    from fairseq.data.data_utils import collate_tokens

    # 前面已经载入了,这里不需要了
    # roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    # roberta.eval()

    batch_of_pairs = [
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
        ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
        ['potatoes are awesome.', 'I like to run.'],
        ['Mars is very far from earth.', 'Mars is very close.'],
    ]

    batch = collate_tokens(
        [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
    )

    logprobs = roberta.predict('mnli', batch)
    print(logprobs.argmax(dim=1))
    # tensor([0, 2, 1, 0]) #为什么是三分类?因为元模型默认已经定义好了3 classes?



    # TODO train example: Use RoBERTa for Filling masks tasks:

    # a sample for fill_mask forward
    # input: The first Star wars movie came out in <mask>
    # <mask>的逻辑要自己写, 归属于数据预处理节点(应该有很多代码可以参考)
    a = roberta.fill_mask('The first Start wars movie came out in <mask>', topk=3)  # 好像一次只能mask一个单词，这边的实现方式
    # [('The first Star wars movie came out in 1977', 0.9504708051681519, ' 1977'), ('The first Star wars movie came out in 1978', 0.009986862540245056, ' 1978'), ('The first Star wars movie came out in 1979', 0.009574787691235542, ' 1979')]
    # 怎么跟住梯度 @jinhui 0313 需要增加RobertaHubInterface接口的操作方法




    # 运用上面的逻辑已经可以直接书写训练代码了,但是实际上RobBERT也提供了训练的接口
    # 不知道是否还是保持梯度跟踪的???




    pass

# exampel()

# 读取一个做MASK的模型，看看能不能继续做SC问题


# TODO 读取一个SC问题的模型，看看他能不能做MASK问题
# 原本的能，但是自己的好像不太可以 @jinhui 0315 应该需要在MASK 上面增加head, 再做sc问题
# 还是会被去掉，需要看一下mnli是如何保持head来finetuning的 估计是保存的方式不同

import torch
# Load the model in fairseq file #@jinhui 会重新修改开源文件
from fairseq.models.roberta import RobertaModel
checkPath1 = "/home/cike/project/fairseq/extension/RoBERT/pre-train/checkpoints"
checkPath2 = "/home/cike/project/fairseq/fairseq_cli/checkpoints/checkpoint1.pt"
# label = 1
pairSentence1 = "CLS I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE"
# label = 0
pairSentence2 = "CLS Since Eve took the first bite of the apple from the tree of knowledge truth in its purity has become a weapon through distortions and promises costing those seeking promised peace great expense and in some cases even the highest prices of all  the lives of our loved ones Anyone who is seeking truth  not just a quick fix or a patch job  will find Dr Patten's book Truth Knowledge or Just Plain Bull a valuable tool in their arsenal of discernment as we are constantly being bombarded with lies and promises to fix your every woe if you will just buy this product belong to this church or political party follow this guideline live here or there etc It is not a book for the faint of heart because I found myself caught in many of the traps hoping to find truth and the peace it promises in so many of the wrong ideas feelings places and products From what I understand from Patten the search for truth is an honorable one Truth does exist But truth is free and its fruit is peace  Not war anarchy chaos hatred or distain for others Hard to imagine something so valuable costing nothing and you don`t even have to drink Jones' Kool-Aid  The issue raised is the cost of trust was a heavy one  it is just that the bill has already been paid in full almost 2000 years ago SPE My brother is Cameron Fry and it wasn't until his first year in college that he found his Ferris Bueller This is a classic that every high school student should see It's a great way to appreciate life and the parade scene is one of the best It's easy to tell from this film that Matthew Broderick was destined to be a star SPE"

roberta = RobertaModel.from_pretrained(checkPath2, checkpoint_file='checkpoint1.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

# Encode a pair of sentences and make a prediction
tokens = roberta.encode(pairSentence1)
a1 = roberta.predict('dsp_head', tokens).argmax()  # 1

# Encode another pair of sentences
tokens = roberta.encode(pairSentence2)
a2 = roberta.predict('dsp_head', tokens).argmax()  # 0


# fill task
roberta_mask = RobertaModel.from_pretrained(checkPath1, checkpoint_file='checkpoint1.pt')
a3 = roberta.fill_mask('The first Start wars movie came out in <mask>', topk=3)



