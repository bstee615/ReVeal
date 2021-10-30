import os

import argparse
import json
import logging
import pickle
from pathlib import Path

import tqdm
from gensim.models import Word2Vec

from data_processing.create_ggnn_data import get_ggnn_graph
from data_processing.create_ggnn_input import read_input, get_input_files
from data_processing.extract_graph import get_graph
from data_processing.extract_slices import get_slices
from split_data import save_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger()

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path, encoding='utf-8', errors='ignore') as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def get_shards(output_dir):
    old_shard_filenames = []
    shard_idx = 0
    shard_filename = output_dir / f'preprocessed_shard{shard_idx}.pkl'
    while shard_filename.exists():
        old_shard_filenames.append(shard_filename)
        shard_idx += 1
        shard_filename = output_dir / f'preprocessed_shard{shard_idx}.pkl'
    return old_shard_filenames, shard_filename



def preprocess(input_dir, wv_path):
    assert input_dir.exists(), input_dir
    files = get_input_files(input_dir)
    assert len(files) > 0, 'no input files'
    logger.info(f'Number of Input Files: {len(files)}')

    # Prepare inputs
    code_dir = input_dir / 'raw_code'
    parsed_dir = input_dir / 'parsed'
    assert code_dir.exists(), f'{code_dir} does not exist'
    assert parsed_dir.exists(), f'{parsed_dir} does not exist'

    # Load previous progress
    all_output_data = []
    model = Word2Vec.load(str(wv_path))

    # Preprocess data
    input_data = read_input(files)
    output_data = []
    for d in tqdm.tqdm(input_data, total=len(files)):
        file_name = d["file_name"]
        label = d["label"]
        i = d["idx"]

        # Sanity checks
        assert file_name == file_name.strip()
        assert label in (0, 1)

        d['file_path'] = str(code_dir / file_name)

        data_instance = dict(d)
        output_data.append(data_instance)

        # Load Joern output
        nodes_file_path = parsed_dir / file_name / 'nodes.csv'
        edges_file_path = parsed_dir / file_name / 'edges.csv'

        # Graph data
        graph_data = get_ggnn_graph(nodes_file_path, edges_file_path, label, model)
        if graph_data is None:
            logger.debug(f'Skipping node {i} ({file_name}) because graph_data is None')
            continue
        data_instance.update(graph_data)
    return all_output_data


def main(cmd_args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--project', help='name of project for differentiating files',
    #                     choices=['chrome_debian', 'devign'], required=True)
    parser.add_argument('-i', '--input', help='input directory, containing <name>/{raw_code,parsed}', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='output and intermediate processing directory', required=True)
    parser.add_argument('--shard_len', help='shard length', type=int, default=5000)
    args = parser.parse_args(cmd_args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(str(output_dir / 'preprocess.log')))

    wv_path = next(Path(args.input[0]).glob('raw_code_*.100'))

    input_dirs = []
    for input_path in args.input:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_dir = Path(input_path)
        assert input_dir.exists(), input_dir
        input_dirs.append(input_dir)

    all_preprocessed_data = {}
    for input_path in args.input:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_dir = Path(input_path)
        assert input_dir.exists(), input_dir
        preprocessed_dir = input_dir / 'preprocessed'
        preprocessed_dir.mkdir(exist_ok=True)

        preprocessed_path = preprocessed_dir / 'preprocessed.pkl'
        if preprocessed_path.exists():
            logger.info(f'loading from {preprocessed_path}...')
            with open(preprocessed_path, 'rb') as f:
                preprocessed_data = pickle.load(f)
        else:
            logger.info(f'preprocessing {input_dir}...')
            preprocessed_data = preprocess(input_dir, wv_path)
            if len(preprocessed_data) > 0:
                logger.info(f'saving to {preprocessed_path}...')
                with open(preprocessed_path, 'wb') as f:
                    pickle.dump(preprocessed_data, f)

        logger.info(f'{len(preprocessed_data)} samples from {input_dir}')
        if input_path in all_preprocessed_data:
            max_id = 1
            for p in all_preprocessed_data.keys():
                try:
                    id = int(p[len(input_path):])
                except ValueError:
                    id = 1
                if id > max_id:
                    max_id = id
            max_id = str(max_id + 1)
            logger.info(f'Already read {input_path}, assuming you want to add it again as augmentation.'
                        f'Adding as {input_path + max_id}')
            all_preprocessed_data[input_path + max_id] = preprocessed_data
        else:
            all_preprocessed_data[input_path] = preprocessed_data

    with open(output_dir / 'info.json', 'w') as f:
        info = {
            "args.input": args.input,
            "wv_name": wv_path.name,
            "len(all_preprocessed_data)": {k: len(v) for k, v in all_preprocessed_data.items()},
        }
        json.dump(info, f, indent=2)

    save_dataset(all_preprocessed_data, output_dir)


if __name__ == '__main__':
    main()
