import copy
import os

import numpy as np
import sys
import torch
from graph_dataset import DataSet
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
from tqdm import tqdm
from tsne import plot_embedding

from models import MetricLearningModel
import logging
logger = logging.getLogger(__name__)


def train(model, dataset, optimizer, save_path, num_epochs, max_patience=5,
          valid_every=1, cuda_device=-1):
    logger.info('Start Training')
    assert isinstance(model, MetricLearningModel) and isinstance(dataset, DataSet)
    best_f1 = 0
    best_model = None
    patience_counter = 0
    train_losses = []
    try:
        for epoch_count in range(num_epochs):
            batch_losses = []
            num_batches = dataset.initialize_train_batches()
            output_batches_generator = range(num_batches)
            for _ in output_batches_generator:
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                features, targets, same_class_features, diff_class_features = dataset.get_next_train_batch()
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                    targets = targets.cuda(device=cuda_device)
                    same_class_features = same_class_features.cuda(device=cuda_device)
                    diff_class_features = diff_class_features.cuda(device=cuda_device)
                probabilities, representation, batch_loss = model(
                    example_batch=features, targets=targets,
                    positive_batch=same_class_features, negative_batch=diff_class_features
                )
                batch_losses.append(batch_loss.detach().cpu().item())
                batch_loss.backward()
                optimizer.step()
            epoch_loss = np.sum(batch_losses).item()
            train_losses.append(epoch_loss)
            logger.info('=' * 100)
            logger.info('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss))
            logger.info('=' * 100)
            if epoch_count % valid_every == 0:
                valid_batch_count = dataset.initialize_valid_batches()
                vacc, vpr, vrc, vf1 = evaluate(
                    model, dataset.get_next_valid_batch, valid_batch_count, cuda_device)
                if vf1 > best_f1:
                    best_f1 = vf1
                    patience_counter = 0
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if False and dataset.initialize_test_batches() != 0:
                    tacc, tpr, trc, tf1 = evaluate(
                        model, dataset.get_next_test_batch, dataset.initialize_test_batches(), cuda_device
                    )
                    logger.info('Test Set:       Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % \
                          (tacc, tpr, trc, tf1))
                    logger.info('=' * 100)
                logger.info('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f\tPatience: %2d' % \
                      (vacc, vpr, vrc, vf1, patience_counter))
                logger.info('-' * 100)
                if patience_counter == max_patience:
                    if best_model is not None:
                        model.load_state_dict(best_model)
                        if cuda_device != -1:
                            model.cuda(device=cuda_device)
                    break
    except KeyboardInterrupt:
        logger.warning('Training Interrupted by User!')
        if best_model is not None:
            model.load_state_dict(best_model)
            if cuda_device != -1:
                model.cuda(device=cuda_device)
    if os.path.exists(save_path):
        _save_file = open(save_path, 'wb')
    else:
        _save_file = open('temp-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    if False and dataset.initialize_test_batches() != 0:
        tacc, tpr, trc, tf1 = evaluate(
            model, dataset.get_next_test_batch, dataset.initialize_test_batches(), cuda_device)
        logger.info('*' * 100)
        logger.info('Test Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % (tacc, tpr, trc, tf1))
        logger.info('%f\t%f\t%f\t%f' % (tacc, tpr, trc, tf1))
        logger.info('*' * 100)

def predict(model, iterator_function, _batch_count, cuda_device):
    probs = predict_proba(model, iterator_function, _batch_count, cuda_device)
    return np.argmax(probs, axis=-1)


def predict_proba(model, iterator_function, _batch_count, cuda_device, inf=False):
    model.eval()
    with torch.no_grad():
        predictions = []
        file_names = []
        if inf:
            for _ in tqdm(range(_batch_count)):
                features, targets, _file_names = iterator_function()
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                probs, _, _ = model(example_batch=features)
                predictions.extend(probs)
                file_names.extend(_file_names)
        else:
            for _ in tqdm(range(_batch_count)):
                features, targets = iterator_function()
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                probs, _, _ = model(example_batch=features)
                predictions.extend(probs)
        model.train()
    if inf:
        return np.array(predictions), file_names
    else:
        return np.array(predictions)


def evaluate(model, iterator_function, _batch_count, cuda_device):
    logger.info(f'Batch: {_batch_count}')
    model.eval()
    with torch.no_grad():
        predictions = []
        expectations = []
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            features, targets = iterator_function()
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            probs, _, _ = model(example_batch=features)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()
            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
        model.train()
        return acc(expectations, predictions) * 100, \
               pr(expectations, predictions) * 100, \
               rc(expectations, predictions) * 100, \
               f1(expectations, predictions) * 100,


def show_representation(model, iterator_function, _batch_count, cuda_device, name):
    model.eval()
    with torch.no_grad():
        representations = []
        expected_targets = []
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            iterator_values = iterator_function()
            features, targets = iterator_values[0], iterator_values[1]
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            _, repr, _ = model(example_batch=features)
            repr = repr.detach().cpu().numpy()
            logger.debug(repr.shape)
            representations.extend(repr.tolist())
            expected_targets.extend(targets.numpy().tolist())
        model.train()
        logger.debug(np.array(representations).shape)
        logger.debug(np.array(expected_targets).shape)
        plot_embedding(representations, expected_targets, title=name)
