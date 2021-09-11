import pickle
import re

import numpy as np
import sklearn.utils

np.random.seed(0)
import logging

logger = logging.getLogger(__name__)

def split_and_save_augmented(input_data, orig_output_dir, splits=(0.7, 0.1, 0.2)):

    assert sum(splits) == 1.0, 'Splits must sum up to 1.0'

    basic_data, augmented_data = input_data["data/chrome_debian"], input_data["data/chrome_debian_refactored2"]

    basic_data = sorted(basic_data, key=lambda d: int(re.sub(r'[^0-9]', '', d["file_name"])))
    augmented_data = sorted(augmented_data, key=lambda d: int(re.sub(r'[^0-9]', '', d["file_name"])))

    logger.info(str([d["file_name"] for d in basic_data[:5]]))
    logger.info(str([d["file_name"] for d in augmented_data[:5]]))

    basic_i, aug_i = 0, 0
    while basic_i < len(basic_data) and aug_i < len(augmented_data):
        basic_fidx = basic_data[basic_i]["file_name"]
        aug_fidx = augmented_data[aug_i]["file_name"]
        while basic_fidx != aug_fidx:
            while basic_fidx < aug_fidx:
                logger.info(f'deleting basic filename {basic_data[basic_i]["file_name"]} (compared to {augmented_data[aug_i]["file_name"]}) at index {basic_i}')
                del basic_data[basic_i]
                basic_fidx = basic_data[basic_i]["file_name"]
            while aug_fidx < basic_fidx:
                logger.info(f'deleting augmented filename {augmented_data[aug_i]["file_name"]} (compared to {basic_data[basic_i]["file_name"]}) at index {aug_i}')
                del augmented_data[aug_i]
                aug_fidx = augmented_data[aug_i]["file_name"]
        basic_i += 1
        aug_i += 1

    logger.info(f'keys: {basic_data[0].keys()}')
    logger.info(str([d["file_name"] for d in basic_data[:5]]))
    logger.info(str([d["file_name"] for d in augmented_data[:5]]))

    assert len(basic_data) == len(augmented_data), f'arrays are not the same size: {len(basic_data)} {len(augmented_data)}'
    basic_data, augmented_data = sklearn.utils.shuffle(basic_data, augmented_data, random_state=0)

    n_folds = 5
    for fold in range(0, n_folds):
        roll = len(basic_data)//n_folds*fold
        logger.info(f'fold {fold} starts at index {roll}')
        cur_basic_data = np.roll(basic_data, roll)
        cur_augmented_data = np.roll(augmented_data, roll)
        cur_output_dir = orig_output_dir / f'fold_{fold}' / 'ggnn_input'

        buggy, non_buggy = [], []
        for example, augmented_example in zip(cur_basic_data, cur_augmented_data):
            assert example["file_name"] == augmented_example["file_name"], f'ordering does not match! {example["file_name"]}, {augmented_example["file_name"]}'
            target = example['targets'][0][0]
            if target == 1:
                buggy.append((example, augmented_example))
            else:
                non_buggy.append((example, augmented_example))

        num_bug = len(buggy)
        num_non_bug = len(non_buggy)
        non_buggy_selected = non_buggy[:num_non_bug]

        train_examples = []
        valid_examples = []
        test_examples = []

        num_train_bugs = int(num_bug * splits[0])
        num_valid_bug = int(num_bug * splits[1])
        train_examples.extend([b[0] for b in buggy[:num_train_bugs]] + [b[1] for b in buggy[:num_train_bugs]])
        valid_examples.extend([b[0] for b in buggy[num_train_bugs:(num_train_bugs + num_valid_bug)]])
        test_examples.extend([b[0] for b in buggy[(num_train_bugs + num_valid_bug):]])

        num_non_bug = len(non_buggy_selected)
        num_train_nobugs = int(num_non_bug * splits[0])
        num_valid_nobug = int(num_non_bug * splits[1])
        train_examples.extend([b[0] for b in non_buggy_selected[:num_train_nobugs]] + [b[1] for b in non_buggy_selected[:num_train_nobugs]])
        valid_examples.extend([b[0] for b in non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)]])
        test_examples.extend([b[0] for b in non_buggy_selected[(num_train_nobugs + num_valid_nobug):]])

        logger.info(f'total: {num_bug + num_non_bug}')
        logger.info(f'train: {len(train_examples)} ({len(train_examples) / (num_bug + num_non_bug) * 100:.2f}%)')
        logger.info(f'valid: {len(valid_examples)} ({len(valid_examples) / (num_bug + num_non_bug) * 100:.2f}%)')
        logger.info(f'test: {len(test_examples)} ({len(test_examples) / (num_bug + num_non_bug) * 100:.2f}%)')

        train_filenames = set(d["file_name"] for d in train_examples)
        valid_filenames = set(d["file_name"] for d in valid_examples)
        test_filenames = set(d["file_name"] for d in test_examples)
        logger.info(f'unique filenames: train: {len(train_filenames)} valid: {len(valid_filenames)} test: {len(test_filenames)}')
        assert len(train_filenames.intersection(test_filenames)) == 0, f'train/test should not overlap: {train_filenames.intersection(test_filenames)}'
        assert len(train_filenames.intersection(valid_filenames)) == 0, f'train/valid should not overlap: {train_filenames.intersection(valid_filenames)}'
        assert len(test_filenames.intersection(valid_filenames)) == 0, f'test/valid should not overlap: {test_filenames.intersection(valid_filenames)}'

        cur_output_dir.mkdir(parents=True, exist_ok=True)
        for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
            f_name = cur_output_dir / (n + '_GGNNinput.pkl')
            logger.info(f'Saving to {f_name}')
            with open(f_name, 'wb') as fp:
                pickle.dump(examples, fp)
                fp.close()



def split_and_save(input_data, output_dir, splits=(0.7, 0.1, 0.2)):

    assert sum(splits) == 1.0, 'Splits must sum up to 1.0'

    input_data = input_data["data/chrome_debian"]
    input_data = sorted(input_data, key=lambda d: int(re.sub(r'[^0-9]', '', d["file_name"])))

    logger.info(f'keys: {input_data[0].keys()}')
    logger.info(str([d["file_name"] for d in input_data[:5]]))

    np.random.shuffle(input_data)

    n_folds = 5
    for fold in range(0, n_folds):
        roll = len(input_data)//n_folds*fold
        logger.info(f'fold {fold} starts at index {roll}')
        cur_input_data = np.roll(input_data, roll)
        cur_output_dir = output_dir / f'fold_{fold}' / 'ggnn_input'

        buggy, non_buggy = [], []
        for example in cur_input_data:
            target = example['targets'][0][0]
            if target == 1:
                buggy.append(example)
            else:
                non_buggy.append(example)

        num_bug = len(buggy)
        num_non_bug = len(non_buggy)
        non_buggy_selected = non_buggy[:num_non_bug]

        train_examples = []
        valid_examples = []
        test_examples = []

        num_train_bugs = int(num_bug * splits[0])
        num_valid_bug = int(num_bug * splits[1])
        train_examples.extend(buggy[:num_train_bugs])
        valid_examples.extend(buggy[num_train_bugs:(num_train_bugs + num_valid_bug)])
        test_examples.extend(buggy[(num_train_bugs + num_valid_bug):])

        num_non_bug = len(non_buggy_selected)
        num_train_nobugs = int(num_non_bug * splits[0])
        num_valid_nobug = int(num_non_bug * splits[1])
        train_examples.extend(non_buggy_selected[:num_train_nobugs])
        valid_examples.extend(non_buggy_selected[num_train_nobugs:(num_train_nobugs + num_valid_nobug)])
        test_examples.extend(non_buggy_selected[(num_train_nobugs + num_valid_nobug):])

        logger.info(f'total: {num_bug + num_non_bug}')
        logger.info(f'train: {len(train_examples)} ({len(train_examples) / (num_bug + num_non_bug) * 100:.2f}%)')
        logger.info(f'valid: {len(valid_examples)} ({len(valid_examples) / (num_bug + num_non_bug) * 100:.2f}%)')
        logger.info(f'test: {len(test_examples)} ({len(test_examples) / (num_bug + num_non_bug) * 100:.2f}%)')

        cur_output_dir.mkdir(parents=True, exist_ok=True)
        for n, examples in zip(['train', 'valid', 'test'], [train_examples, valid_examples, test_examples]):
            f_name = cur_output_dir / (n + '_GGNNinput.pkl')
            logger.info(f'Saving to {f_name}')
            with open(f_name, 'wb') as fp:
                pickle.dump(examples, fp)
                fp.close()
