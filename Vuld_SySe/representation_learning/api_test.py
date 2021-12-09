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
# from sklearn.model_selection import train_test_split
# from baseline_svm import SVMLearningAPI

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger()

def load_train_valid_test(ds, fold, parts):
    """
    Fixed train/valid/test split, preserving the split from GGNN model training.
    """
    features = []
    targets = []
    for part in parts:
        data_file = open(ds + part + f'_GGNNoutput_graph-{fold}.pkl', 'rb')
        logger.info(f'Loading data file {data_file}')
        data = pickle.load(data_file)
        data_file.close()
        features.append([d['graph_feature'] for d in data])
        targets.append([d['target'] for d in data])
        del data

    train_X, valid_X, test_X = map(numpy.array, features)
    train_Y, valid_Y, test_Y = map(numpy.array, targets)
    logger.info(' '.join(str(x) for x in ('train:', train_X.shape, train_Y.shape, 'valid:', valid_X.shape, valid_Y.shape, 'test:', test_X.shape, test_Y.shape)))
    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y


def main(raw_args=None):
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
    parser.add_argument('--num_layers', default=1, help='Bit of a misnomer, but this means 1 internal layer and 1 each input/output layers, so 3 total', type=int)
    parser.add_argument('--num_repeats', default=1, type=int)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--model_dir', type=str, required=True, help='Output file for the best model')
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--n_folds', default=5, type=int)
    parser.add_argument('--no_balance', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args(raw_args)

    logger.info(f'{__name__} args: {args}')
    if args.test:
        logger.info("Quitting because it's just a test.")
        return

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

    output_file_name = os.path.join(args.model_dir, f'{feature_name}-results.tsv')

    n_folds = args.n_folds
    with open(output_file_name, 'w') as output_file:
        for i in range(n_folds):
            model = RepresentationLearningModel(
                lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, max_patience=5, balance=not args.no_balance,
                num_layers=args.num_layers
            )
            logger.info(f'Fold: {i}')
            save_path = os.path.join(args.model_dir, f'{feature_name}-model-{i}.pth')
            # Split data
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = load_train_valid_test(ds, i, parts)
            logger.info('=' * 100)

            model.train(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, save_path, ray=args.ray)
            results = model.evaluate(test_X, test_Y)
            print('Test:', results['accuracy'], results['precision'], results['recall'], results['f1'], flush=True,
                  file=output_file)

            del model

if __name__ == '__main__':
    main()