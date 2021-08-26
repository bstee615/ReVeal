import json
import pickle

import numpy as np

np.random.seed(0)
import logging
from utils import get_shards

logger = logging.getLogger(__name__)


def split_and_save(output_dir):
    shards, _ = get_shards(output_dir)
    input_data = []
    for sh in shards:
        with open(sh, 'rb') as f:
            input_data += pickle.load(f)
    buggy, non_buggy = [], []
    for example in input_data:
        target = example['targets'][0][0]
        if target == 1:
            buggy.append(example)
        else:
            non_buggy.append(example)

    np.random.shuffle(buggy)
    np.random.shuffle(non_buggy)
    num_bug = len(buggy)
    num_non_bug = len(non_buggy)
    non_buggy_selected = non_buggy[:num_non_bug]

    train_examples = []
    valid_examples = []
    test_examples = []

    num_train_bugs = int(num_bug * 0.70)
    num_valid_bug = int(num_bug * 0.10)
    train_examples.extend(buggy[:num_train_bugs])
    valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
    test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])

    num_non_bug = len(non_buggy_selected)
    num_train_nobugs = int(num_non_bug * 0.70)
    num_valid_nobug = int(num_non_bug * 0.10)
    train_examples.extend(non_buggy_selected[:num_train_nobugs])
    valid_examples.extend(non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
    test_examples.extend(non_buggy_selected[(num_train_nobugs + num_valid_nobug):])

    logger.info(f'total: {num_bug + num_non_bug}')
    logger.info(f'train: {len(train_examples)} ({len(train_examples) / (num_bug + num_non_bug) * 100:.2f%)}')
    logger.info(f'valid: {len(valid_examples)} ({len(valid_examples) / (num_bug + num_non_bug) * 100:.2f%)}')
    logger.info(f'test: {len(test_examples)} ({len(test_examples) / (num_bug + num_non_bug) * 100:.2f%)}')

    for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
        f_name = output_dir / (n + '_GGNNinput.json')
        logger.info(f'Saving to {f_name}')
        with open(f_name, 'w') as fp:
            json.dump(examples, fp)
            fp.close()
