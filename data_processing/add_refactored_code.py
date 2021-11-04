"""
Load shard data for testing.
"""
import platform

import os
from multiprocessing import Pool
from pathlib import Path
import pickle
import pandas as pd
import argparse
import tqdm
import tempfile
import shutil
import subprocess

system = platform.system().lower()
old_joern_dir = Path('cfactor/old-joern')  # May have to change to make it match environment
windows = system == 'windows'
if windows:
    jars = [
        old_joern_dir / "projects/extensions/joern-fuzzyc/build/libs/joern-fuzzyc.jar",
        old_joern_dir / 'projects/extensions/jpanlib/build/libs/jpanlib.jar',
    ]
    octopus_jars = list((old_joern_dir / 'projects/octopus/lib').glob('*.jar'))
    assert len(octopus_jars) > 0, 'no jars in old-joern/projects/octopus/lib'
    jars += octopus_jars
    sep = ';' if os.name == 'nt' else ':'
    jars_str = sep.join(str(j) for j in jars)
else:
    joern_parse = Path(__file__).parent.parent / "cfactor/old-joern/joern-parse"


def run_joern(joern_dir, src_dir, src_files=None):
    if windows:
        cmd = f'java ' \
              f'-cp "{jars_str}" ' \
              f'tools.parser.ParserMain ' \
              f'-outformat csv ' \
              f'-outdir {joern_dir} ' \
              f'{src_dir}'
    else:
        cmd = f'bash {joern_parse} -outformat csv -outdir {joern_dir} {src_dir}'
    status = {
        "failed": 0,
        "succeeded": 0,
        "warnings": 0,
    }
    if src_files is None:
        pbar = tqdm.tqdm()
    else:
        pbar = tqdm.tqdm(total=len(src_files))
    with pbar as pbar:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        for line in proc.stdout:
            if 'skipping' in line.lower():
                pbar.update(1)
                status["failed"] += 1
            elif line.strip() in src_files:
                pbar.update(1)
                status["succeeded"] += 1
            elif 'warning' in line:
                status["warnings"] += 1
            pbar.set_postfix(status)
    returncode = proc.wait()
    if returncode != 0:
        print(f'returncode={returncode}')
        # raise Exception(f'Error running command: {cmd}. Last output: {line}')
        print(f'Error running command: {cmd}. Last output: {line}')
    print('Done parsing', src_dir, 'to', joern_dir)


def main(cmd_args=None):
    print('Args:', cmd_args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--nproc', type=int, default=15)
    args = parser.parse_args(cmd_args)

    if args.output_dir is None:
        args.output_dir = args.input_dir
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    # Load new code shards
    shard_idx = 0
    shard_filename = args.input_dir / f'new_functions.pkl.shard{shard_idx}'
    data = []
    while shard_filename.exists():
        with open(shard_filename, 'rb') as f:
            shard = pickle.load(f)
            data.extend(shard)
        shard_idx += 1
        shard_filename = args.input_dir / f'new_functions.pkl.shard{shard_idx}'
    print(len(data))

    raw_code_dir = args.output_dir / 'raw_code'
    raw_code_dir.mkdir(exist_ok=True, parents=True)

    all_parsed_dir = args.output_dir / 'parsed'
    all_filenames = []
    for idx, filename, code, applied in data:
        dst_filepath = raw_code_dir / filename
        with open(dst_filepath, 'w') as f:
            f.write(code)
        all_filenames.append(filename)
    if all_parsed_dir.exists():
        shutil.rmtree(all_parsed_dir)
    run_joern(all_parsed_dir, raw_code_dir, all_filenames)
    result_dir = all_parsed_dir / str(raw_code_dir)
    for child in result_dir.iterdir():
        if child.is_dir():
            shutil.move(str(child), str(all_parsed_dir))


if __name__ == '__main__':
    main()
