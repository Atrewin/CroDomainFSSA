# CroDomainFSSA

### Pycharm Syn, Change the Authority 
> chmod  -R 777 DACroDomainFSSA


### Requirements
* `pytorch = 1.4.0`  
> Needed for Adversarial Training \
> pip install torch==1.4.0
* `transformers==3.0.2`  
> pip install transformers==3.0.2

### 运行graph training 
* 'stanza'
* 'torch_scatter==2.0.5'
* 'torch_sparse==0.6.8'
* 'torch_cluster==1.5.8'
* 'torch_geometric==1.6.3'

* 推荐使用使用whl安装(匹配好pytorch)

* scipy==2.3.4
* gensim==3.8.1

### 运行post training
* [install fairseq](https://github.com/pytorch/fairseq)

### Data Pre-Processing
#### 预处理数据文件
* 将原始情感分析文本数据，转化为json结构数据，方便构建meta-task方式训练。保存在"data/"文件夹下面。
* 运行命令：
> sh data_process.sh
* 预处理得出四个不同领域的json文件以及对应的unsupervised data,分别对应四个领域, keys分别为两个类别："pos" 和 "neg"
> 1. book_reviews.json
> 2. dvd_reviews.json
> 3. Electronics_reviews.json
> 4. kitchen_reviews.json
* 产生8个json预处理后的文件
> 1. book_reviews.json
> 2. book_unlabeled_reviews.json
> 3. dvd_reviews.json
> 4. dvd_unlabeled_reviews.json
> 5. electronics_reviews.json
> 6. electronics_unlabeled_reviews.json
> 7. kitchen_reviews.json
> 8. kitchen_unlabeled_reviews.json

#### 数据文件统计
> python data_statistic.py --json_url <数据json文件的相对路径>


### Run Model
* Proto(CNN)
> CUDA_VISIBLE_DEVICES=5 python train_demo.py --K 5 --Q 5 --pretrain_step 0 --encoder cnn --train book_reviews  --val dvd_reviews --test dvd_reviews --adv dvd_unlabeled_reviews
* Proto(BERT)
> CUDA_VISIBLE_DEVICES=0,1,2 python train_demo.py --K 5 --Q 1 --pretrain_step 100 --encoder bert --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --adv dvd_unlabeled_reviews --batch_size 2
* Proto(Post-BERT)
> CUDA_VISIBLE_DEVICES=0,1,2 python train_demo.py --K 5 --Q 1 --pretrain_step 1000 --encoder roberta --hidden_size 768 --train book_reviews  --val dvd_reviews --test dvd_reviews --adv dvd_unlabeled_reviews --batch_size 2
