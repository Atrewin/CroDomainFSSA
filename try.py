#TODO Load RoBERTa from torch.hub (PyTorch >= 1.1):
import torch
import traceback

try:

    pass
except:
    print(traceback.print_exc())
    pass

def exampel():
    import torch
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()  # disable dropout (or leave in train mode to finetune)

    # Load the model in fairseq file #@jinhui 会重新修改开源文件
    # from fairseq.models.roberta import RobertaModel
    # roberta = RobertaModel.from_pretrained('./pre-train/roberta.large', checkpoint_file='model.pt')
    # roberta.eval()  # disable dropout (or leave in train mode to finetune)

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
def checkPostBert():
    # TODO 读取一个SC问题的模型，看看他能不能做MASK问题
    # 原本的能，但是自己的好像不太可以 @jinhui 0315 应该需要在MASK 上面增加head, 再做sc问题
    # 还是会被去掉，需要看一下mnli是如何保持head来finetuning的 估计是保存的方式不同
    def DSP():
        roberta = RobertaModel.from_pretrained(checkPath1, checkpoint_file='checkpoint1.pt')
        roberta.eval()  # disable dropout (or leave in train mode to finetune)

        # Encode a pair of sentences and make a prediction
        tokens = roberta.encode(pairSentence1)
        a1 = roberta.predict('dsp_head', tokens).argmax()  # 1

        # Encode another pair of sentences
        tokens = roberta.encode(pairSentence2)
        a2 = roberta.predict('dsp_head', tokens).argmax()  # 0

    import torch
    # Load the model in fairseq file #@jinhui 会重新修改开源文件
    from fairseq.models.roberta import RobertaModel
    checkPath1 = "/home/cike/project/fairseq/extension/RoBERT/pre-train/checkpoints/MASK_book2kitchen"
    checkPath2 = "/home/cike/project/fairseq/fairseq_cli/checkpoints"
    # label = 1
    pairSentence1 = "CLS I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE"
    # label = 0
    pairSentence2 = "CLS Since Eve took the first bite of the apple from the tree of knowledge truth in its purity has become a weapon through distortions and promises costing those seeking promised peace great expense and in some cases even the highest prices of all  the lives of our loved ones Anyone who is seeking truth  not just a quick fix or a patch job  will find Dr Patten's book Truth Knowledge or Just Plain Bull a valuable tool in their arsenal of discernment as we are constantly being bombarded with lies and promises to fix your every woe if you will just buy this product belong to this church or political party follow this guideline live here or there etc It is not a book for the faint of heart because I found myself caught in many of the traps hoping to find truth and the peace it promises in so many of the wrong ideas feelings places and products From what I understand from Patten the search for truth is an honorable one Truth does exist But truth is free and its fruit is peace  Not war anarchy chaos hatred or distain for others Hard to imagine something so valuable costing nothing and you don`t even have to drink Jones' Kool-Aid  The issue raised is the cost of trust was a heavy one  it is just that the bill has already been paid in full almost 2000 years ago SPE My brother is Cameron Fry and it wasn't until his first year in college that he found his Ferris Bueller This is a classic that every high school student should see It's a great way to appreciate life and the parade scene is one of the best It's easy to tell from this film that Matthew Broderick was destined to be a star SPE"

    # fill task
    # 读到模型
    roberta_mask = RobertaModel.from_pretrained(checkPath1, checkpoint_file='checkpoint_best.pt')

    a3 = roberta_mask.fill_mask('The first Start wars movie came out in <mask>', topk=3)

    pairSentence1 = "CLS I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE"

    test1 = "CLS I bought this DVD because I really adore the Strokes Good music <mask> people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE I bought this DVD because I really adore the Strokes Good music good people good style not so good documentary Whoever made this video should not make anymore at all The strokes were not even in this besides very few pictures & 10 second clips of them & the exclusive interviews were not that exclusive No sound from any of the strokes & no interviews with them I can go on I mean c'mon a documentary of the Strokes should at least have the Strokes in it right SPE"

    pairSentence2 = "CLS Since Eve took the first bite of the apple from the tree of knowledge truth in its purity has become a weapon through distortions and promises costing those seeking promised peace great expense and in some cases even the highest prices of all  the lives of our loved ones Anyone who is seeking truth  not just a quick fix or a patch job  will find Dr Patten's book Truth Knowledge or Just Plain Bull a valuable tool in their arsenal of discernment as we are constantly being bombarded with lies and promises to fix your every woe if you will just buy this product belong to this church or political party follow this guideline live here or there etc It is not a book for the faint of heart because I found myself caught in many of the traps hoping to find truth and the peace it promises in so many of the wrong ideas feelings places and products From what I understand from Patten the search for truth is an honorable one Truth does exist But truth is free and its fruit is peace  Not war anarchy chaos hatred or distain for others Hard to imagine something so valuable costing nothing and you don`t even have to drink Jones' Kool-Aid  The issue raised is the cost of trust was a heavy one  it is just that the bill has already been paid in full almost 2000 years ago SPE My brother is Cameron Fry and it wasn't until his first year in college that he found his Ferris Bueller This is a classic that every high school student should see It's a great way to appreciate life and the parade scene is one of the best It's easy to tell from this film that Matthew Broderick was destined to be a star SPE"

    test2 = "CLS Since Eve took the first bite of the <mask> from the tree of knowledge truth in its purity has become a weapon through distortions and promises costing those seeking promised peace great expense and in some cases even the highest prices of all  the lives of our loved ones Anyone who is seeking truth  not just a quick fix or a patch job  will find Dr Patten's book Truth Knowledge or Just Plain Bull a valuable tool in their arsenal of discernment as we are constantly being bombarded with lies and promises to fix your every woe if you will just buy this product belong to this church or political party follow this guideline live here or there etc It is not a book for the faint of heart because I found myself caught in many of the traps hoping to find truth and the peace it promises in so many of the wrong ideas feelings places and products From what I understand from Patten the search for truth is an honorable one Truth does exist But truth is free and its fruit is peace  Not war anarchy chaos hatred or distain for others Hard to imagine something so valuable costing nothing and you don`t even have to drink Jones' Kool-Aid  The issue raised is the cost of trust was a heavy one  it is just that the bill has already been paid in full almost 2000 years ago SPE My brother is Cameron Fry and it wasn't until his first year in college that he found his Ferris Bueller This is a classic that every high school student should see It's a great way to appreciate life and the parade scene is one of the best It's easy to tell from this film that Matthew Broderick was destined to be a star SPE"

    sentence1 = "CLS This is the perfect gift for the comic book fan who has every comic book Also looks great on a coffee table"
    test1 = "CLS This is the  <mask> gift for the comic book fan who has every comic book Also looks great on a coffee table"
    test2 = "CLS This is the perfect gift for the comic book fan who has every comic book Also looks <mask> on a coffee table"

    sentence2 = "The book looks good"
    test3 = "The book looks <mask>"

    sentence3 = "CLS The scene in the outdoor cafe is delicious and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date SPE The scene in the outdoor cafe is delicious and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date SPE"
    sentence3 = "CLS The scene in the outdoor cafe is delicious and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date"
    test4 = "CLS The scene in the outdoor cafe is <mask> and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date SPE The scene in the outdoor cafe is delicious and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date SPE"

    sentence3_1 = "CLS The scene in the outdoor cafe is delicious and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date"
    test4_1 = "CLS The scene in the outdoor cafe is <mask> and the scene in the fountain is now a standard  see Boxing Helena  Fellini hit the big-time with this movie and helped develop the future of film A classic - never goes out of date"

    sentence4 = "Works great!  Makes small amounts of delicious coffee!  Very handy"
    test5 = "Works great!  Makes small amounts of <mask> coffee!  Very handy"

    answer1 = roberta_mask.fill_mask(test1, topk=5)  #
    answer2 = roberta_mask.fill_mask(test2, topk=5)  #
    answer3 = roberta_mask.fill_mask(test3, topk=5)  #
    answer4 = roberta_mask.fill_mask(test4, topk=5)  #
    answer5 = roberta_mask.fill_mask(test5, topk=5)  #
    answer4_1 = roberta_mask.fill_mask(test4_1, topk=5)
    print("  ")


def demoForStanza():

    import sys
    import argparse
    import os
    import stanza
    from stanza.resources.common import DEFAULT_MODEL_DIR
    # import stanfordnlp # stanfordnlp 已经被迁移到了stanza
    # from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR

    if __name__ == '__main__':
        # get arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_resources',
                            default=DEFAULT_MODEL_DIR)
        parser.add_argument('-l', '--lang', help='Demo language',
                            default="en")
        parser.add_argument('-c', '--cpu', action='store_true', help='Use cpu as the device.')
        args = parser.parse_args()

        example_sentences = {"en": "Barack Obama was born in Hawaii.  He was elected president in 2008.",
                             "zh": "達沃斯世界經濟論壇是每年全球政商界領袖聚在一起的年度盛事。",
                             "fr": "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie. Il tente d'abord de faire carrière comme marchand d'art chez Goupil & C.",
                             "vi": "Trận Trân Châu Cảng (hay Chiến dịch Hawaii theo cách gọi của Bộ Tổng tư lệnh Đế quốc Nhật Bản) là một đòn tấn công quân sự bất ngờ được Hải quân Nhật Bản thực hiện nhằm vào căn cứ hải quân của Hoa Kỳ tại Trân Châu Cảng thuộc tiểu bang Hawaii vào sáng Chủ Nhật, ngày 7 tháng 12 năm 1941, dẫn đến việc Hoa Kỳ sau đó quyết định tham gia vào hoạt động quân sự trong Chiến tranh thế giới thứ hai."}

        if args.lang not in example_sentences:
            print(
                f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
            sys.exit(1)

        # download the models
        stanza.download(args.lang, args.models_dir)
        # set up a pipeline
        print('---')
        print('Building pipeline...')
        pipeline = stanza.Pipeline(models_dir=args.models_dir, lang=args.lang, use_gpu=(not args.cpu))
        # process the document
        doc = pipeline(example_sentences[args.lang])
        # access nlp annotations
        print('')
        print('Input: {}'.format(example_sentences[args.lang]))
        print("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))
        print('---')
        print('tokens of first sentence: ')
        doc.sentences[0].print_tokens()
        print('')
        print('---')
        print('dependency parse of first sentence: ')
        doc.sentences[0].print_dependencies()
        print('')

    pass


def demoForGetConcepTriple():
    import stanza
    # stanza.download("en")
    nlp = stanza.Pipeline('en', use_gpu=True)

    review = "I am rather annoyed at this because I spent quite a lot of time detailing why I didn't like one of the books, reasonably, I thought."
    review = "this book is boring. I don't like the book"
    # 获取doc的entities
    def getConceptsAndTriple(review):
        doc = nlp(review)

        concepts = []
        opinionConceptTriples = []
        uposFilterConceptType = ["NOUN", "VERB", "ADJ", "ADV"]  # 是否可以做更多的concept类型过滤 upos tag #用xpos会更加精确
        xposFilterConceptType = ["JJ", "JJR", "JJS", "NNS", "NN", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP",
                                 "VBZ"]

        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.words[0].upos in uposFilterConceptType and token.words[0].xpos in xposFilterConceptType:
                    concepts.append(token.words[0].lemma)  # 不知道是否可以使用词性还原spent -> send

            for dependency in sentence.dependencies:
                # 需要定义什么样的语法关系需要保留下来
                src = dependency[0]
                rel = dependency[1]
                dist = dependency[2]
                # 方案1：保留opinionToken为中心的所有关系
                filterType = ["ADJ"]
                if src.id != 0 and (src.upos != "PUNCT" and dist.upos != "PUNCT"):  # root 是没有辅助学习的
                    if src.upos in filterType or dist.upos in filterType:
                        opinionConceptTriples.append((src.lemma, rel, dist.lemma))


        return concepts, opinionConceptTriples

    concepts, opinionConceptTriples = getConceptsAndTriple(review)

    print("  ")
    pass

if __name__ == "__main__":
    demoForGetConcepTriple()

    pass

import traceback

try:

    pass
except:
    print(traceback.print_exc())
    pass