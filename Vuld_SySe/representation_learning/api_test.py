import argparse
import json
import pickle

import logging
import random

import numpy
import os
import sys
import torch
from representation_learning_api import RepresentationLearningModel
from sklearn.model_selection import train_test_split
from baseline_svm import SVMLearningAPI

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger()

def load_train_valid_test(ds):
    """
    Fixed train/valid/test split, preserving the split from GGNN model training.
    """
    features = []
    targets = []
    for part in parts:
        data_file = open(ds + part + '_GGNNinput_graph.pkl', 'rb')
        data = pickle.load(data_file)
        data_file.close()
        features.append([d['graph_feature'] for d in data])
        targets.append([d['target'] for d in data])
        del data

    train_X, valid_X, test_X = map(numpy.array, features)
    train_Y, valid_Y, test_Y = map(numpy.array, targets)
    logger.info(' '.join(str(x) for x in ('train:', train_X.shape, train_Y.shape, 'valid:', valid_X.shape, valid_Y.shape, 'test:', test_X.shape, test_Y.shape)))
    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--dataset', default='chrome_debian/balanced',
#                        choices=['chrome_debian/balanced', 'chrome_debian/imbalanced', 'chrome_debian', 'devign'])
    parser.add_argument('--dataset', default='chrome_debian/balanced')
    parser.add_argument('--features', default='ggnn', choices=['ggnn', 'wo_ggnn'])
    parser.add_argument('--lambda1', default=0.5, type=float)
    parser.add_argument('--lambda2', default=0.001, type=float)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_balance', action='store_true')
    parser.add_argument('--split_old', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_repeats', default=1, type=int)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--model_dir', type=str, required=True, help='Output file for the best model')
    args = parser.parse_args()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset
    feature_name = args.features
    parts = ['train', 'valid', 'test']
    ds = dataset
    assert os.path.exists(dataset), dataset
    assert isinstance(dataset, str)

    logfile_path = os.path.join(args.model_dir, f'reveal-{feature_name}.log')
    if os.path.exists(logfile_path):
        os.unlink(logfile_path)
    logger.addHandler(logging.FileHandler(logfile_path))

    model = RepresentationLearningModel(
        lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, max_patience=5, balance=True,
        num_layers=args.num_layers
    )

    output_file_name = os.path.join(args.model_dir, f'{feature_name}-results.tsv')

    with open(output_file_name, 'w') as output_file:
        save_path = os.path.join(args.model_dir, f'{feature_name}-model.pth')
        # Split data
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = load_train_valid_test(ds)
        logger.info('=' * 100)

        model.train(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, save_path)
        results = model.evaluate(test_X, test_Y)
        print('Test:', results['accuracy'], results['precision'], results['recall'], results['f1'], flush=True,
              file=output_file)
    pass
