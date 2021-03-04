import argparse
import json
import numpy
import os
import sys
import torch
from representation_learning_api import RepresentationLearningModel
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='name of project to distinguish models', default='chrome_debian'
parser.add_argument('--dataset', help='path to model input data', default='../../data/after_ggnn/')
parser.add_argument('--output', help='path to save model after training', default='models/')
parser.add_argument('--lambda1', help='model hyperparameter', default=0.5, type=float)
parser.add_argument('--lambda2', help='model hyperparameter', default=0.001, type=float)
parser.add_argument('--num_layers', default=1, type=int)
numpy.random.rand(1000)
torch.manual_seed(1000)
args = parser.parse_args()

output_dir = args.output
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

features = []
targets = []
parts = ['train', 'valid', 'test']
for part in parts:
    json_data_file = open(ds + part + '_GGNNinput_graph.json')
    data = json.load(json_data_file)
    json_data_file.close()
    for d in data:
        features.append(d['graph_feature'])
        targets.append(d['target'])
    del data
X = numpy.array(features)
Y = numpy.array(targets)
print('Dataset', X.shape, Y.shape, numpy.sum(Y), sep='\t', file=sys.stderr)
print('=' * 100, file=sys.stderr, flush=True)

for _ in range(30):
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, sep='\t', file=sys.stderr, flush=True)

    model = RepresentationLearningModel(
        lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, print=True, max_patience=5, balance=True,
        num_layers=args.num_layers
    )
    
    model.train(train_X, train_Y)
    results = model.evaluate(test_X, test_Y)
    print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t',
            file=sys.stderr, flush=True, end=('\n' + '=' * 100 + '\n'))

save_path = args.output + args.name + '/keidClassifier-model.bin'
_save_file = open(save_path, 'wb')
torch.save(model.state_dict(), _save_file)
_save_file.close()
