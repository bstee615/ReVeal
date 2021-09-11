"""
Load shard data for testing.
"""

from pathlib import Path
import pickle
import pandas as pd
import argparse
import tqdm
import tempfile
import shutil
import subprocess
from create_ggnn_input import get_input
import refactorings

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

args.output_dir = Path(args.output_dir)

joern_parse = Path(__file__).parent.parent / "cfactor/old-joern/joern-parse"

# Load new code shards
shard_idx = 0
shard_filename = Path(f'new_functions.pkl.shard{shard_idx}')
data = []
while shard_filename.exists():
    with open(shard_filename, 'rb') as f:
        shard = pickle.load(f)
        data.extend(shard)
    shard_idx += 1
    shard_filename = Path(f'new_functions.pkl.shard{shard_idx}')
print(len(data))

with tempfile.TemporaryDirectory(prefix=str(Path.cwd().absolute()) + '/tmp_') as tmpdir:
    tmpdir = Path(tmpdir)
    for idx, filename, code, applied in tqdm.tqdm(data):
        raw_code_dir = args.output_dir / 'raw_code'
        raw_code_dir.mkdir(exist_ok=True)
        new_filepath = raw_code_dir / filename
        if new_filepath.exists():
            continue

        tmp_dir = tmpdir / 'tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        dst_filename = tmp_dir / filename
        with open(dst_filename, 'w') as f:
            f.write(code)

        parsed_dir = tmpdir / 'parsed'

        if parsed_dir.exists():
            shutil.rmtree(parsed_dir)

        try:
            subprocess.check_output(f'bash {joern_parse} {tmp_dir.absolute()} -outdir {parsed_dir.absolute()}', shell=True)
        except subprocess.CalledProcessError as e:
            print('error parsing', filename, e)
            continue
        parsed_dst_dir = parsed_dir / str(dst_filename.absolute())[1:]
        assert parsed_dst_dir.is_dir()

        new_parsed_dir = args.output_dir / 'parsed' / filename
        if new_parsed_dir.exists():
            shutil.rmtree(new_parsed_dir)
        shutil.copytree(parsed_dst_dir, new_parsed_dir)
        shutil.copy(dst_filename, new_filepath)
