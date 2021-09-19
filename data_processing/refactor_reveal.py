#!/usr/bin/env python
# coding: utf-8


import argparse
import difflib
import itertools
import multiprocessing
import pickle
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm as tqdm

from cfactor import refactorings
from data_processing.create_ggnn_input import read_input, get_input_files
import random

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()
# logging.getLogger().addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_dir", help="Input source code files", default='./data/chrome_debian')
parser.add_argument('-o', "--output_dir", help="Output refactored source code files", default='./refactored_code')
parser.add_argument('--nproc', default='detect')
parser.add_argument('--mode', default='gen')
parser.add_argument('--slice', default=None)
parser.add_argument('--test', default=None, type=int)
parser.add_argument('--shard-len', default=5000, type=int)
parser.add_argument('--chunk', default=10, type=int)
parser.add_argument('--style', nargs='+', default=['one_of_each'])
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--shuffle_refactorings', action='store_true')
parser.add_argument('--remainder', action='store_true')
parser.add_argument('--no-save', action='store_true')
parser.add_argument('--clean', action='store_true')
parser.add_argument('--no-new-names', action='store_true')
args = parser.parse_args()

style_args = {
    'one_of_each': 0,
    'k_random': 1,
    'threshold': 2,
}
args.style_type = args.style.pop(0)
assert args.style_type in style_args, f'unknown option --style {args.style[0]}'
assert len(args.style) == style_args[args.style_type], f'expected {style_args[args.style_type]} args for --style {args.style_type}, got {len(args.style)}'

proc = subprocess.run('nproc', capture_output=True)
max_nproc = int(proc.stdout)
if args.nproc == 'detect':
    nproc = max_nproc - 1
else:
    nproc = int(args.nproc)
    assert nproc <= max_nproc

args.output_dir = Path(args.output_dir)
if not args.output_dir.exists():
    args.output_dir.mkdir(parents=True)
logging.getLogger().addHandler(logging.FileHandler(str(args.output_dir / f'refactor_reveal-{args.mode}.log')))

random.seed(args.seed)

if args.no_new_names:
    all_refactorings = list(refactorings.refactorings_without_new_names)
else:
    all_refactorings = list(refactorings.all_refactorings)
if args.shuffle_refactorings:
    random.shuffle(all_refactorings)

logging.info(f'random seed: {args.seed}')
logging.info(f'shuffle refactorings: {args.shuffle_refactorings} {[r.__name__ for r in all_refactorings]}')
logging.info(f'transformation style: {args.style_type} {args.style}')
logging.info(f'nproc: {nproc}/{max_nproc}')

def do_one(t):
    idx, fn = t
    try:
        project = refactorings.TransformationProject(
            fn["file_name"], fn["code"],
            transforms=all_refactorings, picker=refactorings.random_picker,
            style=args.style_type, style_args=args.style
        )
        new_lines, applied = project.apply_all(return_applied=True)
        if new_lines is not None:
            return idx, fn["file_name"], ''.join(new_lines), [f.__name__ for f in applied]
        else:
            logger.warning('idx %d filename %s not transformed', idx, fn["file_name"])
            return idx, fn["file_name"], new_lines, [f.__name__ for f in applied]
    except Exception as e:
        logger.exception('idx %d filename %s had an error', idx, fn["file_name"], exc_info=e)
    finally:
        pass


def filter_functions(df):
    already_done_indices = set()
    existing, _ = get_shards()
    for shard in existing:
        with open(shard, 'rb') as f:
            shard = pickle.load(f)
            for r in shard:
                already_done_indices.add(r[0])
    df = df.drop(index=already_done_indices)
    return df


def get_shards():
    existing_shards = []
    shard_idx = 0
    shard_filename = args.output_dir / f'new_functions.pkl.shard{shard_idx}'
    while shard_filename.exists():
        existing_shards.append(shard_filename)
        shard_idx += 1
        shard_filename = args.output_dir / f'new_functions.pkl.shard{shard_idx}'
    new_shard = shard_filename
    return existing_shards, new_shard


def main():
    if args.clean:
        existing_shards, _ = get_shards()
        logger.info(f'cleaning shards {[str(s) for s in existing_shards]}')
        for s in existing_shards:
            s.unlink()

    input_dir = Path(args.input_dir)
    total = len(list((input_dir / 'raw_code').glob('*')))
    logger.info('%d samples', total)
    cfiles = get_input_files(input_dir)
    if args.test is not None:
        logger.info('cutting to %d samples', args.test)
        cfiles = itertools.islice(cfiles, 0, args.test)
    logger.info(f'reading inputs...')
    raw_code_input = read_input(cfiles)
    if args.mode == 'gen':
        func_it = enumerate(raw_code_input)

        # func_it = df.iterrows()
        if args.slice is not None:
            begin, end = args.slice.split(':')
            begin, end = int(begin), int(end)
            func_it = itertools.islice(func_it, begin, end)

        shard_len = args.shard_len

        logger.info('nproc: %d', nproc)

        def save_shard(data):
            if len(data) > 0 and not args.no_save:
                _, new_shard = get_shards()
                with open(new_shard, 'wb') as f:
                    pickle.dump(data, f)

        new_functions = []
        with multiprocessing.Pool(nproc) as p:
            with tqdm.tqdm(total=total) as pbar:
                # For very long iterables using a large value for chunksize can make the job complete
                # much faster than using the default value of 1.
                for new_func in p.imap_unordered(do_one, func_it, args.chunk):
                    new_functions.append(new_func)
                    pbar.update(1)
                    if len(new_functions) >= shard_len:
                        save_shard(new_functions)
                        new_functions = []
                save_shard(new_functions)
    elif args.mode == 'diag':

        new_functions = []

        old_shards, _ = get_shards()
        logger.info('%d shards', len(old_shards))
        for shard in old_shards:
            with open(shard, 'rb') as f:
                shard_new_functions = pickle.load(f)
                logger.info('%d functions in shard %s', len(shard_new_functions), shard)
                new_functions.extend(shard_new_functions)
        logger.info('%d functions total', len(new_functions))
        if len(new_functions) == 0:
            logger.error('0 functions. Quitting.')
            exit(1)

        function_paths = list(cfiles)

        showed = 0
        show_n_programs = 0

        logger.info('counting diffs...')
        applied_counts = defaultdict(int)  # Count of how many programs had a transformation applied
        num_applied = []  # Number of transformations applied to each program
        num_lines = []  # Number of (non-blank) lines in each program's code
        num_blank_lines = []  # Number of (non-blank) lines in each program's code
        num_changed_lines = []  # Number of changed lines in each program's code
        total_switches = 0  # Number of programs containing "switch"
        total_loops = 0  # Number of programs containing "for"
        for i, filename, new_code, applied in tqdm.tqdm(new_functions):
            with open(function_paths[i]) as f:
                old_code = f.read()
            old_lines = old_code.splitlines(keepends=True)
            new_lines = new_code.splitlines(keepends=True)
            diff = difflib.ndiff(old_lines, new_lines)
            for a in applied:
                applied_counts[a] += 1
            num_applied.append(len(applied))
            num_changed_lines.append(len([line for line in diff if line[:2] in ('- ', '+ ')]))
            num_lines.append(len([line for line in old_lines if not line.isspace()]))
            num_blank_lines.append(len([line for line in old_lines if line.isspace()]))

            if 'switch' in old_code:
                total_switches += 1
            if 'for' in old_code:
                total_loops += 1

            if showed < show_n_programs:
                print(f'Applied: {applied}')
                print(''.join(difflib.unified_diff(old_lines, new_lines, fromfile=filename, tofile=filename)))
                showed += 1

            del old_code
            del old_lines
            del new_lines

        for print_function in (logger.info, print):
            # averages
            print_function(f'Average # lines in program: {sum(num_lines) / len(new_functions):.2f}')
            print_function(f'Average # blank lines: {sum(num_blank_lines) / len(new_functions):.2f}')
            print_function(f'# programs with switch: {total_switches}')
            print_function(f'# programs with for loop: {total_loops}')

            print_function(f'Average # changed lines: {sum(num_changed_lines) / len(new_functions):.2f}')
            print_function(f'Average # transformations applied: {sum(num_applied) / len(new_functions):.2f}')
            # totals
            for transform in sorted(all_refactorings, key=lambda r: r.__name__):
                print_function(f'{transform.__name__}: {applied_counts[transform.__name__]}')
            for v in set(num_applied):
                print_function(f'{v} transformations applied: {len([x for x in num_applied if x == v])}')


if __name__ == '__main__':
    main()
