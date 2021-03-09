import argparse
import json
import numpy
import os
import sys
import torch
from graph_dataset import DataSet
from models import MetricLearningModel
from representation_learning_api import RepresentationLearningModel
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='path to classifier model to be used', default='models/FULL/Classifier-model.bin')
parser.add_argument('--dataset', help='path to model input data', default='../inf_test/after_ggnn/FULL/')
parser.add_argument('--output_dir', help='location to place data after ggnn processing', default='results/')
parser.add_argument('--name', help='name of folder to save data in (to differentiate sets)', default='FULL')
args = parser.parse_args()

output_dir = args.output_dir + args.name + '/'

features = []
targets = []
file_names = []
parts = ['train', 'valid', 'test']
for part in parts:
    try:
        json_data_file = open(args.dataset + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
            file_names.append(d['file_name'])
        del data
    except:
        continue
X = numpy.array(features)
Y = numpy.array(targets)
Z = numpy.array(file_names)
print('Dataset', X.shape, Y.shape, numpy.sum(Y), sep='\t', file=sys.stderr)
print('=' * 100, file=sys.stderr, flush=True)

state_dict = torch.load(args.model)
_model = MetricLearningModel(input_dim=X.shape[1], hidden_dim=256)
_model.load_state_dict(state_dict, strict=False)
_model = _model.to('cuda:0')

classifier = RepresentationLearningModel(batch_size=128, print=True, max_patience=5)
classifier.model = _model
classifier.dataset = DataSet(classifier.batch_size, X.shape[1], inf=True)
print('Data & models Loaded')
print('='*83)

output, file_names_out = classifier.predict_proba(X, file_names=Z, inf=True)
output = [e.cpu().tolist() for e in output]
out = [{'result':o, 'file_name':fn} for o, fn in zip(output, file_names_out)]
print('DONE')

with open(args.output_dir + 'finalResults.json', 'w') as of:
    json.dump(out, of, indent=2)
    of.close()
