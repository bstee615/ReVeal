"""
Load shard data for testing.
"""
from multiprocessing import Pool
from pathlib import Path
import pickle
import pandas as pd
import argparse
import tqdm
import tempfile
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', default=None)
parser.add_argument('--nproc', type=int, default=15)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.input_dir
args.input_dir = Path(args.input_dir)
args.output_dir = Path(args.output_dir)

joern_parse = Path(__file__).parent.parent / "cfactor/old-joern/joern-parse"

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
raw_code_dir.mkdir(exist_ok=True)

all_parsed_dir = args.output_dir / 'parsed'
all_parsed_dir.mkdir(exist_ok=True)


def do_one(datum):
    idx, filename, code, applied = datum
    new_filepath = raw_code_dir / filename
    if new_filepath.exists():
        return 'skipped'

    tmp_dir = tmpdir / f'tmp_{filename}'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    dst_filename = tmp_dir / filename
    with open(dst_filename, 'w') as f:
        f.write(code)

    parsed_dir = tmpdir / f'parsed_{filename}'

    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)

    proc = subprocess.run(f'bash {joern_parse} {tmp_dir.absolute()} -outdir {parsed_dir.absolute()}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    if proc.returncode != 0 or 'error' in proc.stdout.lower():
        print(f'error parsing {filename}. Dumping output...')
        print(''.join('> ' + s for s in proc.stdout.splitlines(keepends=True)))
        return 'error'
    parsed_dst_dir = parsed_dir / str(dst_filename.absolute())[1:]
    if not parsed_dst_dir.is_dir():
        print(f'no directory for {filename} ({parsed_dst_dir})')
        return 'error'

    new_parsed_dir = all_parsed_dir / filename
    if new_parsed_dir.exists():
        shutil.rmtree(new_parsed_dir)
    shutil.copytree(parsed_dst_dir, new_parsed_dir)
    shutil.copy(dst_filename, new_filepath)

    shutil.rmtree(parsed_dir)
    shutil.rmtree(tmp_dir)
    return 'success'


with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    with Pool(args.nproc) as p:
        results = {
            "skipped": 0,
            "error": 0,
            "success": 0,
        }
        pbar = tqdm.tqdm(p.imap_unordered(do_one, data, chunksize=10), total=len(data))
        pbar.set_postfix(results)
        for result in pbar:
            results[result] += 1
            pbar.set_postfix(results)
