import sys

sys.path.append('Devign')
import subprocess
import argparse

parser = argparse.ArgumentParser()
# refactored_devign_noname_threshold0.5
parser.add_argument("-n", "--name", help="Name of the run", required=True)
parser.add_argument("-i", "--input_dir", help="Input dataset", required=True)
# 'refactor_reveal', 'add_refactored_code', 'preprocess', 'Devign-preprocess', 'train'
parser.add_argument("--stages", help="Stages to run", default=[], nargs='+')
parser.add_argument("--preprocess_override", help="Refactored input directories for preprocess", default=[], nargs='+')
args = parser.parse_args()

devign_source = args.input_dir
refactored_output = f'out/{args.name}'
preprocessed_output = f'data/{args.name}'

print('Name:', args.name)
print('Stages:', args.stages)

current_stage = None
try:
    if 'refactor_reveal' in args.stages:
        current_stage = 'refactor_reveal'
        print('Stage:', current_stage)
        from data_processing import refactor_reveal
        refactor_args = ''
        if 'noname' in args.name:
            refactor_args += ' --no-new-names'
        if 'threshold0.75' in args.name:
            refactor_args += ' --style threshold 0.75 10'
        if 'buggyonly' in args.name:
            refactor_args += ' --buggy_only'
        seed_arg = next((s for s in args.name.split('_') if 'seed' in s), None)
        if seed_arg is not None:
            refactor_args += ' --shuffle_refactorings --seed ' + seed_arg.split('-')[-1]
        cmd = f'-i {devign_source} -o {refactored_output}{refactor_args}'
        print('Running:', cmd)
        refactor_reveal.main(cmd.split())
    if 'add_refactored_code' in args.stages:
        current_stage = 'add_refactored_code'
        print('Stage:', current_stage)
        from data_processing import add_refactored_code
        add_refactored_code.main(f'--input_dir {refactored_output} --output_dir {refactored_output}'.split())
    if 'preprocess' in args.stages:
        current_stage = 'preprocess'
        print('Stage:', current_stage)
        from data_processing import preprocess
        if args.preprocess_override:
            preprocess_input = ' '.join(args.preprocess_override)
        else:
            preprocess_input = refactored_output
        preprocess.main(f'--input {args.input_dir} {preprocess_input} --output {preprocessed_output}'.split())
    if 'Devign-preprocess' in args.stages:
        current_stage = 'Devign-preprocess'
        print('Stage:', current_stage)
        from Devign import main as devign_main
        devign_main.main(f'--input_dir {preprocessed_output} --preprocess_only'.split())
    if 'train' in args.stages:
        current_stage = 'train'
        print('Stage:', current_stage)
        cmd = f'bash run_model.sh devign {preprocessed_output}'
        print('Command:', cmd)
        subprocess.check_call(cmd, shell=True)
except Exception:
    print('Errored stage:', current_stage)
    raise
