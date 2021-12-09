import logging

import os

import sys
sys.path.append('Devign')

import subprocess
import argparse


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


default_stages = [
    'refactor_reveal', 'add_refactored_code', 'preprocess', 'Devign-preprocess', 'train'
]


def main():
    # Define args
    parser = argparse.ArgumentParser()
    parser.add_argument("--refactor-options", help="Options for refactoring", nargs='*')
    parser.add_argument("--model", help="Which model to train", choices=['devign', 'reveal'])
    parser.add_argument("--input_dir", help="Input dataset", required=True)
    parser.add_argument("--output_dir", help="Where to output results", default='output')
    parser.add_argument("--skip", help="Stages to skip", choices=default_stages, nargs='*')
    parser.add_argument("--vanilla", action='store_true', help='No refactoring')
    parser.add_argument('--buggy_only_augmented', help='only add buggy samples when augmenting', action='store_true')

    parser.add_argument("--n_folds", type=int)
    parser.add_argument("--ray", action='store_true')
    args = parser.parse_args()

    print(__name__, 'args:', args)

    # Postprocess args
    if args.skip is None:
        stages_to_skip = []
    else:
        stages_to_skip = args.skip
    if args.refactor_options is None:
        refactor_options = []
    else:
        refactor_options = args.refactor_options
    input_dir = args.input_dir
    vanilla = args.vanilla

    # Make output directories
    if vanilla:
        stages_to_skip += ['refactor_reveal', 'add_refactored_code']
        name = '_'.join([os.path.basename(input_dir), *refactor_options, 'vanilla'])
    else:
        name = '_'.join([os.path.basename(input_dir), *refactor_options])
    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    refactored_output = os.path.join(output_dir, 'refactored_pickle')
    preprocessed_output = os.path.join(output_dir, 'preprocessed_output')
    stages = [stage for stage in default_stages if stage not in stages_to_skip]

    model = args.model

    # Set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(os.path.join(output_dir, 'combined.log'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Start
    print(f'Name: {name}')
    print(f'Stages to skip: {stages_to_skip}')
    print(f'Stages to run: {stages}')
    for stage in stages:
        print(f'begin stage: {stage}')
        try:
            do_stage(stage, model, input_dir, output_dir, refactored_output, preprocessed_output, refactor_options,
                     vanilla, args.buggy_only_augmented, args.ray)
        except Exception:
            print(f'errored stage: {stage}')
            raise


def do_stage(stage, model_type, input_dir, output_dir, refactored_output_dir, preprocessed_output_dir, refactor_options,
             vanilla, buggy_only_augmented=False, ray=False, test=False):
    if test:
        print(f'DO STAGE {stage}')
        return
    if stage == 'refactor_reveal':
        do_refactor_reveal(input_dir, refactor_options, refactored_output_dir)
    if stage == 'add_refactored_code':
        do_add_refactored_code(output_dir, refactored_output_dir)
    if stage == 'preprocess':
        do_preprocess(input_dir, output_dir, preprocessed_output_dir, vanilla, buggy_only_augmented)
    if stage == 'Devign-preprocess':
        do_Devign_preprocess(preprocessed_output_dir)
    if stage == 'train':
        do_train(model_type, preprocessed_output_dir, ray)
    print(f'end stage: {stage}')


def do_train(model_type, preprocessed_output_dir, ray):
    if model_type == 'both':
        cmd = f'bash batch_model.sh reveal {preprocessed_output_dir} {"--ray" if ray else ""}'
        print(f'Command: {cmd}')
        subprocess.check_call(cmd, shell=True)
        cmd = f'bash batch_model.sh devign {preprocessed_output_dir} {"--ray" if ray else ""}'
        print(f'Command: {cmd}')
        subprocess.check_call(cmd, shell=True)
    else:
        cmd = f'bash batch_model.sh {model_type} {preprocessed_output_dir} {"--ray" if ray else ""}'
        print(f'Command: {cmd}')
        subprocess.check_call(cmd, shell=True)


def do_Devign_preprocess(preprocessed_output_dir):
    from Devign import main as devign_main
    cmd = f'--input_dir {preprocessed_output_dir} --preprocess_only'
    print(f'Command: {cmd}')
    devign_main.main(cmd.split())


def do_preprocess(input_dir, output_dir, preprocessed_output_dir, vanilla, buggy_only_augmented):
    from data_processing import preprocess
    if vanilla:
        cmd = f'--input {input_dir} --output {preprocessed_output_dir} {"--buggy_only_augmented" if buggy_only_augmented else ""}'
    else:
        cmd = f'--input {input_dir} {output_dir} --output {preprocessed_output_dir} {"--buggy_only_augmented" if buggy_only_augmented else ""}'
    print(f'Command: {cmd}')
    preprocess.main(cmd.split())


def do_add_refactored_code(output_dir, refactored_output_dir):
    from data_processing import add_refactored_code
    cmd = f'--input_dir {refactored_output_dir} --output_dir {output_dir}'
    print(f'Command: {cmd}')
    add_refactored_code.main(cmd.split())


def do_refactor_reveal(input_dir, refactor_options, refactored_output_dir):
    from data_processing import refactor_reveal
    refactor_args = []
    if 'style' in refactor_options:
        style_value = refactor_options[refactor_options.index('style') + 1]
        refactor_args.append(f'--style {style_value}')
    if 'no-new-names' in refactor_options:
        refactor_args.append('--no-new-names')
    if 'threshold' in refactor_options:
        threshold_value = refactor_options[refactor_options.index('threshold') + 1]
        refactor_args.append(f'--style threshold {threshold_value} 10')
    if 'buggonly' in refactor_options:
        refactor_args.append('--buggy_only')
    if 'seed' in refactor_options:
        seed_value = refactor_options[refactor_options.index('seed') + 1]
        refactor_args.append('--shuffle_refactorings')
        refactor_args.append(f'--seed {seed_value}')
    cmd = f'-i {input_dir} -o {refactored_output_dir} {" ".join(refactor_args)} --clean'
    print(f'Command: {cmd}')
    refactor_reveal.main(cmd.split())


if __name__ == '__main__':
    main()
