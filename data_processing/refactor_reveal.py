#!/usr/bin/env python
# coding: utf-8
import os

import re

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

import tqdm as tqdm

from cfactor import refactorings
from data_processing.create_ggnn_input import read_input, get_input_files
import random

import logging

# logger = logging.getLogger()
# logger.addHandler(logging.StreamHandler())

def do_one(t):
    (idx, fn), (args, all_refactorings) = t
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
            print('WARNING: idx %d filename %s not transformed', idx, fn["file_name"])
            return idx, fn["file_name"], new_lines, [f.__name__ for f in applied]
    except Exception as e:
        print('ERROR: idx %d filename %s had an error', idx, fn["file_name"], exc_info=e)
    finally:
        pass


def get_shards(output_dir):
    existing_shards = []
    shard_idx = 0
    shard_filename = output_dir / f'new_functions.pkl.shard{shard_idx}'
    while shard_filename.exists():
        existing_shards.append(shard_filename)
        shard_idx += 1
        shard_filename = output_dir / f'new_functions.pkl.shard{shard_idx}'
    new_shard = shard_filename
    return existing_shards, new_shard


def main(cmd_args=None):
    # logging.getLogger().addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_dir", help="Input source code files")
    parser.add_argument('-o', "--output_dir", help="Output refactored source code files", default='./refactored_code')
    parser.add_argument('-p', "--read_pickle_dir", help="Read pickles from an alternative directory")
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
    parser.add_argument('--buggy_only', action='store_true')
    parser.add_argument('--no-new-names', action='store_true')
    args = parser.parse_args(cmd_args)

    style_args = {
        'one_of_each': 0,
        'k_random': 1,
        'threshold': 2,
    }
    args.style_type = args.style.pop(0)
    print(f'args.style_type: {args.style_type}')
    assert args.style_type in style_args, f'unknown option --style {args.style_type} ({args.style})'
    assert len(args.style) == style_args[args.style_type], f'expected {style_args[args.style_type]} args for --style {args.style_type}, got {len(args.style)}'

    try:
        proc = subprocess.run('nproc', capture_output=True)
        max_nproc = int(proc.stdout)
        if args.nproc == 'detect':
            nproc = max_nproc - 1
        else:
            nproc = int(args.nproc)
            assert nproc <= max_nproc
    except Exception:
        nproc = 8
        max_nproc = 8

    args.output_dir = Path(args.output_dir)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    # logging.getLogger().addHandler(logging.FileHandler(str(args.output_dir / f'refactor_reveal-{args.mode}.log')))

    if args.read_pickle_dir is not None:
        args.read_pickle_dir = Path(args.read_pickle_dir)

    random.seed(args.seed)

    if args.no_new_names:
        all_refactorings = list(refactorings.refactorings_without_new_names)
    else:
        all_refactorings = list(refactorings.all_refactorings)
    if args.shuffle_refactorings:
        random.shuffle(all_refactorings)

    print(f'random seed: {args.seed}')
    print(f'shuffle refactorings: {args.shuffle_refactorings} {[r.__name__ for r in all_refactorings]}')
    print(f'transformation style: {args.style_type} {args.style}')
    print(f'nproc: {nproc}/{max_nproc}')
    if args.clean:
        existing_shards, _ = get_shards(args.output_dir)
        print(f'cleaning shards {[str(s) for s in existing_shards]}')
        for s in existing_shards:
            s.unlink()

    input_dir = Path(args.input_dir)
    total = len(list((input_dir / 'raw_code').glob('*')))
    print(f'{total} samples')
    cfiles = get_input_files(input_dir)
    if args.test is not None:
        print('cutting to %d samples', args.test)
        cfiles = itertools.islice(cfiles, 0, args.test)
    print(f'reading inputs...')
    raw_code_input = read_input(cfiles, read_code=True)
    if args.mode == 'gen':
        generate(raw_code_input, total, args, nproc, all_refactorings)
    elif args.mode == 'diag':
        diag(cfiles, args.output_dir, all_refactorings, args.read_pickle_dir)


def generate(raw_code_input, total, args, nproc, all_refactorings):
    func_it = enumerate(raw_code_input)
    # func_it = df.iterrows()
    if args.slice is not None:
        begin, end = args.slice.split(':')
        begin, end = int(begin), int(end)
        func_it = itertools.islice(func_it, begin, end)
    shard_len = args.shard_len
    print('nproc: %d', nproc)

    if args.buggy_only:
        func_it = (f for f in func_it if f[1]["label"] == 1)

    def save_shard(data):
        if len(data) > 0 and not args.no_save:
            _, new_shard = get_shards(args.output_dir)
            with open(new_shard, 'wb') as f:
                pickle.dump(data, f)

    new_functions = []
    with multiprocessing.Pool(nproc) as p:
        with tqdm.tqdm(total=total) as pbar:
            # For very long iterables using a large value for chunksize can make the job complete
            # much faster than using the default value of 1.
            for new_func in p.imap_unordered(do_one, ((t, (args, all_refactorings)) for t in func_it), args.chunk):
                new_functions.append(new_func)
                pbar.update(1)
                if len(new_functions) >= shard_len:
                    save_shard(new_functions)
                    new_functions = []
            save_shard(new_functions)


def diag(cfiles, output_dir, all_refactorings, read_pickle_dir=None):
    if read_pickle_dir is not None:
        new_functions = load_old_functions(read_pickle_dir)
    else:
        new_functions = load_old_functions(output_dir)
    if len(new_functions) == 0:
        print('ERROR: No refactored functions. Quitting.')
        exit(1)
    function_paths = list(cfiles)
    showed = 0
    show_n_programs = 3
    print('counting diffs...')
    applied_counts = defaultdict(int)  # Count of how many programs had a transformation applied
    num_applied = []  # Number of transformations applied to each program
    num_blank_lines = []  # Number of blank lines in each program's code
    num_lines = []  # Number of lines in each program's code
    num_changed_lines = []  # Number of changed lines in each program's code
    percent_changed_lines = []  # Percent of lines changed in each program's code
    total_switches = 0  # Number of programs containing "switch"
    total_loops = 0  # Number of programs containing "for"
    total_noswitchnoloop = 0
    for i, filename, new_code, applied in tqdm.tqdm(new_functions):
        with open(next(fp for fp in function_paths if fp.name == filename)) as f:
            old_code = f.read()
        # print(applied)
        for a in set(applied):
            applied_counts[a] += 1
        # num_applied.append(len(applied))
        # TODO: This needs to use unified_diff or something that merges modified lines
        # prog_num_changed_lines = len([line for line in diff if line[:2] in ('- ', '+ ')])

        tmp1, tmp2 = tempfile.mktemp(), tempfile.mktemp()
        try:
            with open(tmp1, 'w') as f:
                f.write(old_code)
            with open(tmp2, 'w') as f:
                f.write(new_code)
            wc_str = subprocess.check_output(f'sdiff --suppress-common-lines {tmp1} {tmp2} | wc -l', shell=True)
            prog_num_changed_lines = int(wc_str)
        finally:
            if os.path.exists(tmp1):
                os.unlink(tmp1)
            if os.path.exists(tmp2):
                os.unlink(tmp2)

        old_lines = old_code.splitlines(keepends=True)

        # new_lines = new_code.splitlines(keepends=True)
        # diff = difflib.ndiff(old_lines, new_lines)
        # diff = difflib.unified_diff(old_lines, new_lines)
        # prog_num_changed_lines = len([line for line in diff if line[0] in ('-', '+') and line[:3] not in ('---', '+++')])

        prog_num_lines = len([line for line in old_lines])
        num_changed_lines.append(prog_num_changed_lines)
        num_lines.append(prog_num_lines)
        num_blank_lines.append(len([line for line in old_lines if line.isspace()]))

        prog_percent_changed_lines = prog_num_changed_lines / prog_num_lines
        percent_changed_lines.append(prog_percent_changed_lines)

        # if 'switch' in old_code:
        #     total_switches += 1
        # if 'for' in old_code:
        #     total_loops += 1
        has_switch = False
        has_loop = False
        if re.search(r'switch\s*\(', old_code, flags=re.MULTILINE):
            total_switches += 1
            has_switch = True
        if re.search(r'for\s*\(', old_code, flags=re.MULTILINE):
            total_loops += 1
            has_loop = True
        if not has_switch and not has_loop:
            total_noswitchnoloop += 1

        if showed < show_n_programs:
            print('Filename:', filename)
            print('Num lines:', prog_num_lines, 'Num changed:', prog_num_changed_lines)
            print(f'Applied: {applied}')
            print('Old code:', old_code)
            print('New code:', new_code)
            # print('Diff:', ''.join(difflib.unified_diff(old_lines, new_lines, fromfile=filename, tofile=filename)))
            showed += 1

        del old_code
        del old_lines
        # del new_lines
    # for print_function in (print, print):
        # averages
        # print_function(f'Average # lines in program: {sum(num_lines) / len(new_functions):.2f}')
        # print_function(f'Average # blank lines: {sum(num_blank_lines) / len(new_functions):.2f}')
        # print_function(f'# programs with switch: {total_switches}')
        # print_function(f'# programs with for loop: {total_loops}')
        #
        # print_function(f'Average # changed lines: {sum(num_changed_lines) / len(new_functions):.2f}')
        # print_function(f'Average # transformations applied: {sum(num_applied) / len(new_functions):.2f}')
        # # totals
        # for transform in sorted(all_refactorings, key=lambda r: r.__name__):
        #     print_function(f'{transform.__name__}: {applied_counts[transform.__name__]}')
        # for v in set(num_applied):
        #     print_function(f'{v} transformations applied: {len([x for x in num_applied if x == v])}')
    # print('Transforms:', applied_counts)
    print('***** Total programs:', len(new_functions))
    print(f'***** # programs with switch: {total_switches}')
    print(f'***** # programs with for loop: {total_loops}')
    print(f'***** # programs with no switch or for loop: {total_noswitchnoloop}')
    for v in sorted(set(applied_counts)):
        print(f'***** {v} transformations applied: {applied_counts[v]}')
    print(f'***** Average # blank lines: {sum(num_blank_lines) / len(new_functions):.2f}')

    print(f'***** Average # nonblank lines in program: {sum(num_lines) / len(new_functions):.2f}')
    print(f'***** Average # changed lines: {sum(num_changed_lines) / len(new_functions):.2f}')
    print(f'***** Average % changed lines (macro average): {sum(num_changed_lines) / sum(num_lines):.2f}')
    print(f'***** Average % changed lines (micro average): {sum(percent_changed_lines) / len(new_functions):.2f}')


def load_old_functions(output_dir):
    new_functions = []
    old_shards, _ = get_shards(output_dir)
    print('%d shards', len(old_shards))
    # if len(old_shards) == 0:
    #     rp_dir = output_dir / 'refactored_pickle'
    #     print(f'trying to load shards from {rp_dir}')
    #     old_shards, _ = get_shards(rp_dir)
    #     print('%d shards', len(old_shards))
    for shard in old_shards:
        with open(shard, 'rb') as f:
            shard_new_functions = pickle.load(f)
            print('%d functions in shard %s', len(shard_new_functions), shard)
            new_functions.extend(shard_new_functions)
    print('%d functions total', len(new_functions))
    return new_functions


if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #                     datefmt='%Y-%m-%d:%H:%M:%S',
    #                     level=print)
    main()
