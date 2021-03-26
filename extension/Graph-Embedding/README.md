# KinGDOM: Knowledge-Guided DOMain adaptation for sentiment analysis (ACL 2020)

[_KinGDOM_](https://arxiv.org/abs/2005.00791.pdf) takes a novel perspective on the task of domain adaptation in sentiment analysis by exploring the role of external commonsense knowledge. It utilizes the ConceptNet knowledge graph to enrich the semantics of a document by providing both domain-specific and domain-general background concepts. These concepts are learned by training a graph convolutional autoencoder that leverages inter-domain concepts in a domain-invariant manner. Conditioning a popular domain-adversarial baseline method with these learned concepts helps improve its performance over state-of-the-art approaches, demonstrating the efficacy of the proposed framework.

![Alt text](KinGDOM.jpeg?raw=true "KinGDOM framework")

### Requirements
- scipy==1.3.1
- gensim==3.8.1
- torch==1.6.0
- numpy==1.18.2
- scikit_learn==0.22.2.post1
- torch_geometric==1.6.3

### Execution

Download ConceptNet filtered for English language from [here](https://drive.google.com/file/d/19klcp69OYEf29A_JrBphgkMVPQ9rXe1k/view?usp=sharing) and keep in this root directory.

Preprocess, train and extract graph features:

```bash
python preprocess_graph.py
python train_and_extract_graph_features.py
```

We provide pretrained graph features in the `graph_features` directory. Note that, executing the above commands will overwrite the provided feature files.

Train the main domain adaptation model:

```bash
python train.py
```

Some of the RGCN functionalities are adapted from https://github.com/JinheonBaek/RGCN

### Citation

Please cite the following paper if you find this code useful in your work.

```bash
KinGDOM: Knowledge-Guided DOMain adaptation for sentiment analysis. D. Ghosal, D. Hazarika, N. Majumder, A. Roy, S. Poria, R. Mihalcea. ACL 2020.
```
\mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
\mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
\frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j

\mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
![1](http://latex.codecogs.com/svg.latex?\int_a^bf(x)\ dx)


![1](http://latex.codecogs.com/svg.latex?\mathbf{x}^{\prime}_i=\mathbf{\Theta}_{\textrm{root}}\cdot\mathbf{x}_i+\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_r(i)}\frac{1}{|\mathcal{N}_r(i)|}\mathbf{\Theta}_r\cdot\mathbf{x}_j)

![2](http://latex.codecogs.com/svg.latex?\mathbf{x}_i+\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_r(i)}\frac{1}{|\mathcal{N}_r(i)|}\mathbf{\Theta}_r\cdot\mathbf{x}_j)

![3](http://latex.codecogs.com/svg.latex?a)
![1](http://latex.codecogs.com/svg.latex?\mathbf{x}^{\prime}_i)
![1](http://latex.codecogs.com/svg.latex?\mathbf{x}^{\prime}_i)

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.