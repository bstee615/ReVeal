import json
import os
import sys
import argparse
from tqdm import tqdm
#import clang.cindex
#import clang.enumerations
import csv
import numpy as np
import re 
#from graphviz import Digraph
#import nltk
#from gensim.models import Word2Vec

#try:
#    # set the config
#    clang.cindex.Config.set_library_path("/usr/lib/x86_64-linux-gnu")
#    clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1')
#except:
#    pass   

parser = argparse.ArgumentParser()
parser.add_argument('--project', help='name of project for differentiating files', default='chrome_debian')
#parser.add_argument('--input', help='directory where raw code and parsed are stored', default='../data/chrome_debian')
parser.add_argument('--base', help='paths to loaction of json shards to be ccombined', default='../data/full_experiment_real_data/')
parser.add_argument('--output', help='output directory for resulting json file', default='../data/full_experiment_real_data_processed/')
args = parser.parse_args()

base_dir = args.base
project = args.project
shards = os.listdir(os.path.join(base_dir, project))

def extract_graph_data(
    project, portion, base_dir=args.base, output_dir=args.output):
    assert portion in ['full_graph', 'cgraph', 'dgraph', 'cdgraph']
    shards = os.listdir(os.path.join(base_dir, project))
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(base_dir, project, project + '.json.shard' + str(sc)))
        shard_data = json.load(shard_file)
        for data in tqdm(shard_data):
            fidx = data['id']
            label = int(data['label'])
            total_functions.add(fidx)
            present = data[portion] is not None
            code_graph = data[portion]
            if present:
                code_graph['id'] = fidx
                code_graph['file_name'] = data['file_name']
                code_graph['file_path'] = data['file_path']
                code_graph['code'] = data['code']
                graphs.append(code_graph)
                in_scope_function.add(fidx)
            else:
                if label == 1:
                    vnt += 1
                else:
                    nvnt += 1
        shard_file.close()
        del shard_data
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = open(os.path.join(output_dir, project + '-' + portion + '.json'), 'w')
    json.dump(graphs, output_file)
    output_file.close()
    print(project, portion, len(total_functions), len(in_scope_function), vnt, nvnt, sep='\t')

extract_graph_data(args.project, 'full_graph')


