import pickle
import re

import numpy as np
import sklearn.utils

np.random.seed(0)
import logging

logger = logging.getLogger(__name__)


def split_and_save(data, output_dir, splits=(0.7, 0.1, 0.2), input_dataset=None, augmented_datasets=None, n_folds=5):

    assert sum(splits) == 1.0, 'Splits must sum up to 1.0'

    if input_dataset is None:
        input_dataset = list(data.keys())[0]

    logger.info(f'input data {input_dataset} augmented data {augmented_datasets}')
    input_data = data[input_dataset]
    input_data = sorted(input_data, key=lambda d: int(''.join(s for s in d["file_name"].split('_') if s.isnumeric())))
    logger.info(f'keys: {input_data[0].keys()}')
    logger.info(str([d["file_name"] for d in input_data[:5]]))

    if augmented_datasets is None:
        augmented_datasets = []
    aug_datasets_by_filename = {}
    for dataset_name in augmented_datasets:
        aug_dataset = data[dataset_name]
        logger.info(f'loaded {len(aug_dataset)} augmented samples from {dataset_name}')
        aug_datasets_by_filename[dataset_name] = {d["file_name"]: d for d in aug_dataset}

    np.random.shuffle(input_data)

    if n_folds is not None:
        for fold in range(0, n_folds):
            roll = len(input_data)//n_folds*fold
            logger.info(f'fold {fold} starts at index {roll}')
            cur_input_data = np.roll(input_data, roll)

            cur_output_dir = output_dir / f'fold_{fold}' / 'ggnn_input'

            write_split(cur_input_data, cur_output_dir, splits, aug_datasets_by_filename)
    else:
        cur_output_dir = output_dir / 'ggnn_input'
        write_split(input_data, cur_output_dir, splits, aug_datasets_by_filename)


def write_split(cur_input_data, cur_output_dir, splits, aug_datasets_by_filename):
    buggy, non_buggy = [], []
    for example in cur_input_data:
        target = example['targets'][0][0]
        if target == 1:
            buggy.append(example)
        else:
            non_buggy.append(example)
    num_bug = len(buggy)
    num_non_bug = len(non_buggy)
    train_examples = []
    valid_examples = []
    test_examples = []
    num_train_bugs = int(num_bug * splits[0])
    num_valid_bug = int(num_bug * splits[1])
    train_examples.extend(buggy[:num_train_bugs])
    valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
    test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])
    num_train_nobugs = int(num_non_bug * splits[0])
    num_valid_nobug = int(num_non_bug * splits[1])
    train_examples.extend(non_buggy[:num_train_nobugs])
    valid_examples.extend(non_buggy[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
    test_examples.extend(non_buggy[(num_train_nobugs + num_valid_nobug):])
    aug_examples = []
    for dataset_name, dataset in aug_datasets_by_filename.items():
        matching_examples = [dataset[d["file_name"]] for d in train_examples if d["file_name"] in dataset]
        logger.info(f'adding {len(matching_examples)} augmented samples from {dataset_name}')
        aug_examples.extend(matching_examples)
    train_examples.extend(aug_examples)
    total_outputs = len(train_examples) + len(valid_examples) + len(test_examples)
    logger.info(f'total: {total_outputs}')
    logger.info(
        f'train: {len(train_examples)} ({len(train_examples) / total_outputs * 100:.2f}%) ({len(train_examples) / len(cur_input_data) * 100:.2f}% of original dataset)')
    logger.info(
        f'valid: {len(valid_examples)} ({len(valid_examples) / total_outputs * 100:.2f}%) ({len(valid_examples) / len(cur_input_data) * 100:.2f}% of original dataset)')
    logger.info(
        f'test: {len(test_examples)} ({len(test_examples) / total_outputs * 100:.2f}%) ({len(test_examples) / len(cur_input_data) * 100:.2f}% of original dataset)')
    cur_output_dir.mkdir(parents=True, exist_ok=True)
    for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
        f_name = cur_output_dir / (n + '_GGNNinput.pkl')
        logger.info(f'Saving to {f_name}')
        with open(f_name, 'wb') as fp:
            pickle.dump(examples, fp)
            fp.close()
