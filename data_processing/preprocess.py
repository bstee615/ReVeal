import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import tqdm
from gensim.models import Word2Vec

from data_processing.create_ggnn_data import get_ggnn_graph
from data_processing.create_ggnn_input import read_input, get_input_files
from data_processing.extract_graph import get_graph
from data_processing.extract_slices import get_slices
from split_data import split_and_save

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



def preprocess(input_dir, preprocessed_dir, shard_len, wv_path, portion='full_graph',
               draper=False, vuld_syse=False):
    assert input_dir.exists(), input_dir

    files = get_input_files(input_dir)
    assert len(files) > 0, 'no input files'
    logger.info(f'Number of Input Files: {len(files)}')

    # Prepare inputs
    code_dir = input_dir / 'raw_code'
    parsed_dir = input_dir / 'parsed'
    assert code_dir.exists(), code_dir
    assert parsed_dir.exists(), parsed_dir

    # Load previous progress
    all_output_data = []
    old_shards, _ = get_shards(preprocessed_dir)
    logger.info(f'Loaded {len(old_shards)} shards')
    for shard_filename in old_shards:
        with open(shard_filename, 'rb') as f:
            shard_data = pickle.load(f)
            # logger.info(f'len: {len(shard_data)} min: {min(p["idx"] for p in shard_data)} max: {max(p["idx"] for p in shard_data)}')
            all_output_data += shard_data
    if len(all_output_data) > 0:
        max_idx = max(p["idx"] for p in all_output_data)
        logger.info(f'Skipping to index {max_idx}/{len(files)}, assuming all prior indices are sequential')
    else:
        max_idx = 0
    if max_idx == len(files) - 1:
        return all_output_data

    # Load pretrained Word2Vec model. Might need to be renamed
    # if it's freshly extracted from replication.zip.
    # Paper uses an embedding size of 100
    model = Word2Vec.load(str(wv_path))

    # Preprocess data
    input_data = read_input(files)
    pbar = tqdm.tqdm(total=len(files), desc=f'{input_dir}')
    pbar.update(max_idx)
    # output_data_logged = len(loaded_progress)
    output_data = []
    for d in input_data:
        try:
            file_name = d["file_name"]
            label = d["label"]
            code = d["code"]
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
            nodes = read_csv(nodes_file_path)
            edges = read_csv(edges_file_path)
            if len(nodes) == 0:
                logger.debug(f'Skipping node {i} ({file_name}) because len(nodes) == 0')
                continue

            # Tokenize code
            if draper:
                import clang_stuff
                code_text = ' '.join(code.splitlines(keepends=True))
                t_code = clang_stuff.tokenize(code_text)
                if t_code is None:
                    logger.debug(f'Skipping node {i} ({file_name}) because t_code is None')
                    continue
                data_instance["draper"] = t_code

            # Get slices for VulDeePecker/SySeVR
            if vuld_syse:
                from data_processing.extract_linized_slices import get_linized_slices
                combined_graph = get_graph(nodes, edges)
                vuld_syse_slice_data = get_slices(combined_graph, nodes)
                vuld_syse_line_data = get_linized_slices(code, vuld_syse_slice_data)
                data_instance.update(vuld_syse_line_data)

            # Graph data
            graph_data = get_ggnn_graph(nodes_file_path, edges_file_path, label, model, portion)
            if graph_data is None:
                logger.debug(f'Skipping node {i} ({file_name}) because graph_data is None')
                continue
            data_instance.update(graph_data)

            # Return data instance
            # output_data_logged += 1
            del d

            # Save shards periodically
            if len(output_data) > 0 and len(output_data) % shard_len == 0:
                _, new_shard = get_shards(preprocessed_dir)
                logger.info(f'saving shard {new_shard}')
                with open(new_shard, 'wb') as f:
                    pickle.dump(output_data, f)
                    all_output_data.extend(output_data)
                    del output_data
                    output_data = []
        finally:
            pbar.update(1)
            # pbar.set_postfix({"output data": f'{output_data_logged} ({output_data_logged/total*100:.2f}%)'})
    if len(output_data) > 0:
        _, new_shard = get_shards(preprocessed_dir)
        logger.info(f'saving last shard {new_shard}')
        with open(new_shard, 'wb') as f:
            pickle.dump(output_data, f)
            all_output_data.extend(output_data)
            del output_data
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
    # wv_name = f'raw_code_{args.project}.100'
    # wv_path = Path(args.input[0]) / wv_name

    all_preprocessed_data = {}
    for input_path in args.input:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        logger.info(f'preprocessing {input_path}...')
        input_dir = Path(input_path)
        assert input_dir.exists(), input_dir
        preprocessed_dir = input_dir / 'preprocessed' / input_dir.name
        preprocessed_dir.mkdir(exist_ok=True, parents=True)
        preprocessed_data = preprocess(input_dir, preprocessed_dir, args.shard_len, wv_path)
        logger.info(f'{len(preprocessed_data)} samples from {input_dir}')
        preprocessed_data = [d for d in preprocessed_data if "graph" in d]
        logger.info(f'cut to {len(preprocessed_data)} samples')
        if input_path in all_preprocessed_data:
            logger.info(f'Already read {input_path}, assuming you want to add it again as augmentation. Adding as {input_path + "2"}')
            all_preprocessed_data[input_path + '2'] = preprocessed_data
        else:
            all_preprocessed_data[input_path] = preprocessed_data

    with open(output_dir / 'info.json', 'w') as f:
        info = {
            "args.input": args.input,
            "wv_name": wv_path.name,
            "len(all_preprocessed_data)": {k: len(v) for k, v in all_preprocessed_data.items()},
        }
        json.dump(info, f, indent=2)

    split_and_save(all_preprocessed_data, output_dir)


if __name__ == '__main__':
    main()
