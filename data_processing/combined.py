import os

import sys
sys.path.append('Devign')

import subprocess
import argparse

from Devign import main as devign_main
from data_processing import preprocess
from data_processing import add_refactored_code
from data_processing import refactor_reveal

from datetime import datetime


def print_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

default_stages = [
    'refactor_reveal', 'add_refactored_code', 'preprocess', 'Devign-preprocess', 'train'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", help="Options for the run", nargs='*')
    parser.add_argument("--input_dir", help="Input dataset", required=True)
    parser.add_argument("--stages-to-skip", help="Stages to skip", choices=default_stages, nargs='*')
    args = parser.parse_args()

    if args.stages_to_skip is None:
        stages_to_skip = []
    else:
        stages_to_skip = args.stages_to_skip
    if args.options is None:
        options = []
    else:
        options = args.options
    input_dir = args.input_dir

    name = '_'.join([os.path.basename(input_dir), *options])
    output_dir = os.path.join('output', name)
    os.makedirs(output_dir, exist_ok=True)
    refactored_output = os.path.join(output_dir, 'refactored_pickle')
    # add_refactored_code_output = os.path.join(output_dir, 'refactored_code')
    preprocessed_output = os.path.join(output_dir, 'preprocessed_output')
    stages = [stage for stage in default_stages if stage not in stages_to_skip]

    print('Name:', name)
    print('Stages to skip:', stages_to_skip)
    print('Stages to run:', stages)

    for stage in stages:
        print('begin stage:', stage)
        print_current_time()
        try:
            if stage == 'refactor_reveal':
                refactor_args = []
                if 'no-new-names' in options:
                    refactor_args.append('--no-new-names')
                if 'threshold' in options:
                    threshold_value = options.find('threshold') + 1
                    refactor_args.append(f'--style threshold {threshold_value} 10')
                if 'buggonly' in options:
                    refactor_args.append('--buggy_only')
                if 'seed' in options:
                    seed_value = options.find('seed') + 1
                    refactor_args.append('--shuffle_refactorings')
                    refactor_args.append(f'--seed {seed_value}')
                cmd = f'-i {input_dir} -o {refactored_output} {" ".join(refactor_args)} --clean'
                print('Command:', cmd)
                refactor_reveal.main(cmd.split())
            elif stage == 'add_refactored_code':
                cmd = f'--input_dir {refactored_output} --output_dir {output_dir}'
                print('Command:', cmd)
                add_refactored_code.main(cmd.split())
            elif stage == 'preprocess':
                cmd = f'--input {input_dir} {output_dir} --output {preprocessed_output}'
                print('Command:', cmd)
                preprocess.main(cmd.split())
            elif stage == 'Devign-preprocess':
                cmd = f'--input_dir {preprocessed_output} --preprocess_only'
                print('Command:', cmd)
                devign_main.main(cmd.split())
            elif stage == 'train':
                cmd = f'bash run_model.sh devign {preprocessed_output}'
                print('Command:', cmd)
                subprocess.check_call(cmd, shell=True)
            del cmd
        except Exception:
            print('Errored stage:', stage)
            raise
        finally:
            print('end stage:', stage)
            print_current_time()

if __name__ == '__main__':
    main()
