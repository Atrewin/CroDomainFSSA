# CroDomainFSSA

### Pycharm Syn, Change the Authority 
> chmod  -R 777 DACroDomainFSSA


### Requirements
> Basic
* `pytorch = 1.7.1`  
* `transformers==3.0.2`  

> For graph training 
* `stanza`
* `torch_scatter==2.0.5`
* `torch_sparse==0.6.8`
* `torch_cluster==1.5.8`
* `torch_geometric==1.6.3`
* `scipy==2.3.4`
* `gensim==3.8.1`

> For post-bert training
* [install fairseq](https://github.com/pytorch/fairseq)

### Data Pre-Processing
> 1. Download raw data  
    Download raw data form [here](), then put them in `data/domain_data/init_data`
> 2. Transfer raw data to json format data  
    Run the `data/data_process_utils/data_process.sh`  
    You will get 8 json file in `data/domain_data/processed_data`  
    Notes: this script includes concept extractor, so it will take some time to run 
> 3. Link to [conceptNet](http://conceptnet.io/)  
    Run the python file `data/data_process_utils/data_linkConceptNet`  
    You will get concept triplets for each reviews by adding a new key `conceptNetTriples` to their json file which are got by the previous step  
    Notes: there are a lot of concept to link and conceptNet has rate limits, this will take about five hours.
> 4. Format data for graph training  
    Run the python file `data/data_process_utils/data_genrate_fomat_triple`  
    You will get a conceptnet_english_ours.txt which is needed in domain graph construction  
### Prepare Encoder
    there are tow encoder to train(graph encoder and bert encoder)
#### Train graph encoder and get graph feature  
> 1. This is based on Github project [Kingdom](https://github.com/declare-lab/kingdom)  
> 2. Samply, you can run `extension/Graph-Embedding/preprocess_graph.py` to aggregate a domain graph for each domain
>>  For example, aggregate a domain graph for books domain by running `extension/Graph-Embedding/preprocess_graph.py` with parameter `--domain books`
> 3. Run `extension/Graph-Embedding/train_graph_model.py` with parameter `--domain books`(or other domain name in `{"books", "dvd"，"electronics", "kitchen"}`)  
    The trained GCN model weight of the domain is placed in `extension/Graph-Embedding/weight`  
> 4. Add graph feature data by the key `graphFeature` for each reviews json which have generated at Data Pre-Processing phase  
>> Run `extension/Graph-Embedding/add_grap_feature.py` with `--domain books` (or other in `{"books", "dvd"，"electronics", "kitchen"}`)  

#### Train prost-bert encoder
> 1. This extension is based on fairseq. You can follow [fairseq example](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md) to pretrain bert
>> Samply, you can run `extension/Post-Bert/data/data_process_utils/data_process.sh` to generate fairseq data format.  
>> Then, run `extension/Post-Bert/data/data_process_utils/precess2databin.sh`
> 2. Run `extension/Post-Bert/data/data_process_utils/MASK_training.sh` to get post-bert checkpoint which will be located in `extension/Post-Bert/checkpoints` 
>> Remenber you need to switch domains in `{ book2dvd, book2kitchen, book2electronis, dvd2electronis, dvd2kitchen, electronis2kitchen }`
>> Checkpoint will be loaded when you run DAProNetModel


### Run Model
   You can run train_demo.py to get start. As our paper describes, the model will encoder a review to latent code which is the base for calculating prototype. 
   `--encoder` can be used to select different sentence encoder in `{ cnn, bert, roberta, bert_newGraph, roberta_newGraph, graph}`. 
   `--pretrain_ckpt` is used to load checkpoint for bert base model. The post-bert checkpoints which are trained in Prepare Encoder phase can passed in here. if not given, the basic model will be used meaning post-training strategy is lossed. 
   `--is_old_graph_feature 1` means the graph feature is the same with [Kingdom](https://github.com/declare-lab/kingdom). Before using it, you need to download `graph_features` form Kingdom and put it in `data/`. Notes it only works when `--encoder` in `{bert_newGraph, roberta_newGraph, graph}`
   `--train_iter`, `--val_iter` and `--test_iter` means to do how many iters. 
   

* Proto(CNN)
> CUDA_VISIBLE_DEVICES=5 python train_demo.py --K 5 --Q 5  --encoder cnn --train book_reviews  --val dvd_reviews --test dvd_reviews
* Proto(BERT)
> CUDA_VISIBLE_DEVICES=0,1,2 python train_demo.py --K 5 --Q 2  --encoder bert --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --batch_size 2
* Proto(Post-BERT)
> CUDA_VISIBLE_DEVICES=3,1,2 python train_demo.py --K 5 --Q 2 --encoder roberta --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --batch_size 2 --pretrain_ckpt ${checkpoint}
* Proto(Post-BERT,New-Graph)
> CUDA_VISIBLE_DEVICES=4 python train_demo.py --K 5 --Q 2 --encoder roberta_newGraph --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --batch_size 1 --pretrain_ckpt /home/cike/project/DACroDomainFSSA/extension/Post-Bert/checkpoints/MASK_book2dvd/checkpoint200_d_b_92.6.pt --train_iter 10000 --grad_iter  16
* Proto(Post-BERT,Old-Graph)
> CUDA_VISIBLE_DEVICES=4 python train_demo.py --K 5 --Q 2 --encoder roberta_newGraph --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --batch_size 1 --pretrain_ckpt /home/cike/project/DACroDomainFSSA/extension/Post-Bert/checkpoints/MASK_book2dvd/checkpoint200_d_b_92.6.pt --train_iter 10000 --grad_iter  16 --is_old_graph_feature 1
* Proto(graph)
> > CUDA_VISIBLE_DEVICES=0,1,2 python train_demo.py --K 5 --Q 2  --encoder graph --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --batch_size 2

* 切分实验设计
