import os
import pickle
import re

import numpy as np
import sklearn.utils

np.random.seed(0)
import logging

logger = logging.getLogger(__name__)


def split_and_save(data, output_dir, splits=(0.7, 0.1, 0.2)):

    assert sum(splits) == 1.0, 'Splits must sum up to 1.0'

    dataset_names = list(data.keys())
    input_dataset, augmented_datasets = dataset_names[0], dataset_names[1:]
    logger.info(f'input data {input_dataset} augmented data {augmented_datasets}')

    write_data(data, input_dataset, augmented_datasets, output_dir / 'ggnn_input')


def write_data(data, input_dataset, augmented_datasets, output_dir):
    input_data = data[input_dataset]
    np.random.shuffle(input_data)
    logger.info(f'total: {len(input_data)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'GGNNinput.pkl', 'wb') as fp:
        pickle.dump(input_data, fp)

    all_aug_data = {}
    for aug_dataset in augmented_datasets:
        aug_data = data[aug_dataset]
        logger.info(f'augmentation {aug_dataset}: {len(aug_data)}')
        aug_data = {d["file_name"]:d for d in aug_data}
        all_aug_data[aug_dataset] = aug_data
    with open(output_dir / 'augmented_GGNNinput.pkl', 'wb') as fp:
        pickle.dump(all_aug_data, fp)
