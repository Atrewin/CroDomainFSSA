from tqdm import tqdm
import numpy as np
import os.path, pickle
from utils import obtain_all_seed_concepts
from utils_graph import conceptnet_graph, domain_aggregated_graph, subgraph_for_concept

import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
# 没有起到切换encoding的效果，变成在open（）上改

if __name__ == '__main__':
    
    bow_size = 5000
    
    print ('Extracting seed concepts from all domains.')
    all_seeds = obtain_all_seed_concepts(bow_size)
    
    print ('Creating conceptnet graph.')#是独立的超参数图，应该来源于ConceptNet
    G, G_reverse, concept_map, relation_map = conceptnet_graph('conceptnet_english.txt')
    
    print ('Num seed concepts:', len(all_seeds))
    print ('Populating domain aggregated sub-graph with seed concept sub-graphs.')
    triplets, unique_nodes_mapping = domain_aggregated_graph(all_seeds, G, G_reverse, concept_map, relation_map)# @jinhui 这边是全部dataset的吗？
    
    print ('Creating sub-graph for seed concepts.')
    concept_graphs = {}# 每个seed 一个graph是为什么？在总图中把相关的东西都筛选出来

    for node in tqdm(all_seeds, desc='Instance', position=0):
        concept_graphs[node] = subgraph_for_concept(node, G, G_reverse, concept_map, relation_map)
        
    # Create mappings
    inv_concept_map = {v: k for k, v in concept_map.items()}
    inv_unique_nodes_mapping = {v: k for k, v in unique_nodes_mapping.items()}
    inv_word_index = {}
    for item in inv_unique_nodes_mapping:
        inv_word_index[item] = inv_concept_map[inv_unique_nodes_mapping[item]]
    word_index = {v: k for k, v in inv_word_index.items()}
        
    print ('Saving files.')
        
    pickle.dump(all_seeds, open('preprocess_data/all_seeds.pkl', 'wb'))
    pickle.dump(concept_map, open('preprocess_data/concept_map.pkl', 'wb'))
    pickle.dump(relation_map, open('preprocess_data/relation_map.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open('preprocess_data/unique_nodes_mapping.pkl', 'wb'))
    pickle.dump(word_index, open('preprocess_data/word_index.pkl', 'wb'))
    pickle.dump(concept_graphs, open('preprocess_data/concept_graphs.pkl', 'wb'))#每个concept的独立子图，就是多了成索引
    
    np.ndarray.dump(triplets, open('preprocess_data/triplets.np', 'wb'))        #所用concept的子图
    print ('Completed.')