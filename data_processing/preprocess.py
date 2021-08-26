import argparse
import json
import logging
import pickle
from pathlib import Path

import tqdm
from gensim.models import Word2Vec

from new_data_processing.create_ggnn_data import get_ggnn_graph
from new_data_processing.create_ggnn_input import create_ggnn_input
from new_data_processing.extract_graph import get_graph
from new_data_processing.extract_slices import get_slices
from split_data import split_and_save
from utils import read_csv

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def get_shards(output_dir):
    old_shard_filenames = []
    shard_idx = 0
    shard_filename = output_dir / f'preprocessed_{shard_idx}'
    while shard_filename.exists():
        old_shard_filenames.append(shard_filename)
        shard_idx += 1
        shard_filename = output_dir / f'preprocessed_{shard_idx}'
    return old_shard_filenames, shard_filename


def preprocess(total, input_data, input_dir, output_dir, project, wv_name, shard_len, portion='full_graph',
               draper=False, vuld_syse=False):
    # Prepare inputs
    if isinstance(input_data, Path):
        input_data = json.load(open(input_data))
    input_data = input_data
    pbar = tqdm.tqdm(total=total)
    code_dir = input_dir / 'raw_code'
    parsed_dir = input_dir / 'parsed'
    assert code_dir.exists(), code_dir
    assert parsed_dir.exists(), parsed_dir
    logger.info(f'Number of Input Files: {len(input_data)}')
    model = Word2Vec.load(input_dir / project / wv_name)

    # Load previous progress
    loaded_progress = []
    old_shards, _ = get_shards(output_dir)
    for shard_filename in old_shards:
        loaded_progress += pickle.load(open(shard_filename, 'rb'))

    # Preprocess data
    output_data = loaded_progress
    input_data = input_data[len(loaded_progress):]
    pbar.update(len(loaded_progress))
    for d in input_data:
        file_name = d["file_name"]
        label = d["label"]
        code = d["code"]
        i = d["idx"]

        # Sanity checks
        assert file_name == file_name.strip()
        assert label in (0, 1)

        data_instance = {
            'file_path': str(code_dir / file_name),
            'code': code,
            'label': int(label)
        }

        # Load Joern output
        nodes_file_path = parsed_dir / file_name / 'nodes.csv'
        edges_file_path = parsed_dir / file_name / 'edges.csv'
        nodes = read_csv(nodes_file_path)
        edges = read_csv(edges_file_path)
        if len(nodes) == 0:
            logger.debug(f'Skipping node {i} ({file_name}) because len(nodes) == 0')
            continue

        combined_graph = get_graph(nodes, edges)

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
            from new_data_processing.extract_linized_slices import get_linized_slices
            vuld_syse_slice_data = get_slices(combined_graph, nodes)
            vuld_syse_line_data = get_linized_slices(code, vuld_syse_slice_data)
            data_instance.update(vuld_syse_line_data)

        # Graph data
        graph_data = get_ggnn_graph(nodes_file_path, edges_file_path, label, model, portion)
        if graph_data is None:
            continue
        data_instance.update(graph_data)

        # Return data instance
        output_data.append(data_instance)
        pbar.update(1)
        del d

        # Save shards periodically
        if i % shard_len == 0:
            _, new_shard = get_shards(output_dir)
            with open(new_shard, 'wb') as f:
                pickle.dump(output_data, f)
                del output_data
                output_data = []
    logger.info(len(output_data), 'items')

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files',
                        choices=['chrome_debian', 'devign'], required=True)
    parser.add_argument('--input', help='input directory, containing <name>/{raw_code,parsed}', required=True)
    parser.add_argument('--output', help='output and intermediate processing directory', required=True)
    parser.add_argument('--restart', help='restart processing instead of loading saved shards', action='store_true')
    parser.add_argument('--word2vec_name', help='Word2Vec file name')
    parser.add_argument('--shard_len', help='shard length', type=int, default=5000)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    assert input_dir.exists(), input_dir
    output_dir.mkdir(exist_ok=True)

    total, full_text_files = create_ggnn_input(input_dir, output_dir, args.project)
    preprocessed_data = preprocess(total, full_text_files, input_dir, output_dir, args.project, args.wv_name,
                                   args.shard_len)
    split_and_save(preprocessed_data, output_dir / args.project)
