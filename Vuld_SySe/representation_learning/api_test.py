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


def load_train_valid_test_old(ds):
    """
    Old version of train/valid/test split, which leaks data from
    test to train and has a minor arithmetic bug when splitting train/valid.
    """
    features = []
    targets = []
    for part in parts:
        data_file = open(ds + part + '_GGNNinput_graph.pkl', 'rb')
        data = pickle.load(data_file)
        data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    X = numpy.array(features)
    Y = numpy.array(targets)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=1/8)
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
    parser.add_argument('--baseline_model', default='svm')
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
    # if os.path.exists(dataset):
        # Load path to the dataset
        # ds = dataset
    # else:
        # Load dataset by name
        # if feature_name == 'ggnn':
        #     if dataset == 'chrome_debian/balanced':
        #         ds = '../../data/after_ggnn/chrome_debian/balance/v3/'
        #     elif dataset == 'chrome_debian/imbalanced':
        #         ds = '../../data/after_ggnn/chrome_debian/imbalance/v6/'
        #     elif dataset == 'chrome_debian/repro':
        #         ds = '../../out/data/after_ggnn/chrome_debian/'
        #     elif dataset == 'chrome_debian/partial_data':
        #         ds = '../../data-old/after_ggnn_partial/chrome_debian/'
        #     elif dataset == 'devign':
        #         ds = '../../data/after_ggnn/devign/v6/'
        #     else:
        #         raise ValueError('Imvalid Dataset')
        # else:
        #     if dataset == 'chrome_debian':
        #         ds = '../../data/full_experiment_real_data_processed/chrome_debian/full_graph/v1/graph_features/'
        #     elif dataset == 'devign':
        #         ds = '../../data/full_experiment_real_data_processed/devign/full_graph/v1/graph_features/'
        #     else:
        #         raise ValueError('Imvalid Dataset')
    ds = dataset
    assert os.path.exists(dataset), dataset
    assert isinstance(dataset, str)
    # save_path = os.path.join(args.model_dir, dataset.replace('/', '_') + '-' + feature_name)
    # os.makedirs(save_path, exist_ok=True)
    # if args.split_old:
    #     save_path += '-old_split'
    save_path = os.path.join(args.model_dir, f'{feature_name}-model.pth')

    # output_dir = os.path.join(args.model_dir, dataset.replace('/', '_') + '-' + feature_name, 'results_test')
    # if args.baseline:
    #     output_dir = 'baseline_' + args.baseline_model
    #     if args.baseline_balance:
    #         output_dir += '_balance'
    # os.makedirs(output_dir, exist_ok=True)
    # output_file_name = os.path.join(output_dir, dataset.replace('/', '_') + '-' + feature_name)
    # if args.split_old:
    #     output_file_name = output_file_name + '-old_split'
    # if args.lambda1 == 0:
    #     assert args.lambda2 == 0
    #     output_file_name += 'cross-entropy-only_layers-'+ str(args.num_layers) + '.tsv'
    # else:
    #     output_file_name += 'triplet-loss_layers-'+ str(args.num_layers) + '.tsv'
    output_file_name = os.path.join(args.model_dir, f'{feature_name}-results.tsv')

    logger.addHandler(logging.FileHandler(os.path.join(args.model_dir, f'reveal-{feature_name}.log')))

    with open(output_file_name, 'w') as output_file:
        # Split data
        if args.split_old:
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = load_train_valid_test_old(ds)
        else:
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = load_train_valid_test(ds)
        logger.info('=' * 100)

        if args.baseline:
            model = SVMLearningAPI(True, args.baseline_balance, model_type=args.baseline_model)
        else:
            model = RepresentationLearningModel(
                lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, max_patience=5, balance=True,
                num_layers=args.num_layers
            )
        model.train(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, save_path)
        results = model.evaluate(test_X, test_Y)
        print('Test:', results['accuracy'], results['precision'], results['recall'], results['f1'], flush=True,
              file=output_file)
    pass
