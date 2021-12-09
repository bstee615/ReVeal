import numpy
import sys
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train, predict, predict_proba, evaluate as evaluate_from_model
import logging


logger = logging.getLogger(__name__)


class RepresentationLearningModel(BaseEstimator):
    def __init__(self,
                 alpha=0.5, lambda1=0.5, lambda2=0.001, hidden_dim=256,  # Model Parameters
                 dropout=0.2, batch_size=64, balance=True,   # Model Parameters
                 num_epoch=100, max_patience=20,  # Training Parameters
                 num_layers=1):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dropout = dropout
        self.num_epoch = num_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size
        self.balance = balance
        self.cuda = torch.cuda.is_available()
        assert self.cuda
        self.num_layers = num_layers
        pass

    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y, save_path, ray=False):
        input_dim = train_x.shape[1]
        self.model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=self.hidden_dim, aplha=self.alpha, lambda1=self.lambda1,
            lambda2=self.lambda2, dropout_p=self.dropout, num_layers=self.num_layers
        )
        logger.info(f'Model: {self.model}')
        logger.info(f'Hyperparameters: dropout_p={self.dropout}, aplha={self.alpha}, lambda1={self.lambda1}, lambda2={self.lambda2}, num_layers={self.num_layers}')
        self.optimizer = Adam(self.model.parameters())
        if self.cuda:
            self.model.cuda(device=0)
        self.dataset = DataSet(self.batch_size, train_x.shape[1])
        
        # This code is fishy because it takes 10% of train data, which is already only 80% of the dataset,
        # meaning the train/valid/test split is actually 72/8/20.
        #for _x, _y in zip(train_x, train_y):
        #    if numpy.random.uniform() <= 0.1:
        #        self.dataset.add_data_entry(_x.tolist(), _y.item(), 'valid')
        #    else:
        #        self.dataset.add_data_entry(_x.tolist(), _y.item(), 'train')

        for _x, _y in zip(train_x, train_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), 'train')
        for _x, _y in zip(valid_x, valid_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), 'valid')
        for _x, _y in zip(test_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), 'test')
        self.dataset.initialize_dataset(balance=self.balance)
        train(
            model=self.model, dataset=self.dataset, optimizer=self.optimizer,
            save_path=save_path, num_epochs=self.num_epoch, max_patience=self.max_patience,
            cuda_device=0 if self.cuda else -1, ray=ray
        )
        logger.info('Training Complete')

    def predict(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return predict(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
        )

    def predict_proba(self, text_x, file_names=None, inf=False):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        if inf:
            for _x, _fn in zip(text_x, file_names):
                self.dataset.add_data_entry(_x.tolist(), 0, part='test', file_name=_fn)
            return predict_proba(
                model=self.model, iterator_function=self.dataset.get_next_test_batch,
                _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
                inf=True)

        else:
            for _x in text_x:
                self.dataset.add_data_entry(_x.tolist(), 0, part='test')
            return predict_proba(
                model=self.model, iterator_function=self.dataset.get_next_test_batch,
                _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
            )

    def evaluate(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        acc, pr, rc, f1 = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
        )
        return {
            'accuracy': acc,
            'precision': pr,
            'recall': rc,
            'f1': f1
        }

    def score(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        _, _, _, f1 = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
        )
        return f1