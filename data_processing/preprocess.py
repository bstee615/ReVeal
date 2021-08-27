import argparse
import logging
import pickle
from pathlib import Path

import tqdm
from gensim.models import Word2Vec

from data_processing.create_ggnn_data import get_ggnn_graph
from data_processing.create_ggnn_input import get_input
from data_processing.extract_graph import get_graph
from data_processing.extract_slices import get_slices
from data_processing.utils import get_shards
from split_data import split_and_save
from utils import read_csv

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def preprocess(total, project_dir, output_dir, wv_name, shard_len, portion='full_graph',
               draper=False, vuld_syse=False):
    # Prepare inputs
    pbar = tqdm.tqdm(total=total)
    code_dir = project_dir / 'raw_code'
    parsed_dir = project_dir / 'parsed'
    assert code_dir.exists(), code_dir
    assert parsed_dir.exists(), parsed_dir
    logger.info(f'Number of Input Files: {total}')

    # Load previous progress
    loaded_progress = []
    old_shards, _ = get_shards(output_dir)
    for shard_filename in old_shards:
        with open(shard_filename, 'rb') as f:
            loaded_progress += pickle.load(f)
    if len(loaded_progress) > 0:
        max_idx = loaded_progress[-1]["idx"]
    else:
        max_idx = 0

    # Preprocess data
    output_data = []
    all_output_data = []
    pbar.update(max_idx)
    output_data_logged = len(loaded_progress)
    if max_idx == total - 1:
        return
    input_data = get_input(project_dir, start=max_idx)
    model = Word2Vec.load(str(project_dir / wv_name))
    for d in input_data:
        try:
            file_name = d["file_name"]
            label = d["label"]
            code = d["code"]
            i = d["idx"]

            # Sanity checks
            assert file_name == file_name.strip()
            assert label in (0, 1)

            data_instance = d
            data_instance['file_path'] = str(code_dir / file_name)

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
            output_data.append(data_instance)
            output_data_logged += 1
            del d

            # Save shards periodically
            if len(output_data) > 0 and len(output_data) % shard_len == 0:
                _, new_shard = get_shards(output_dir)
                with open(new_shard, 'wb') as f:
                    pickle.dump(output_data, f)
                    all_output_data.extend(output_data)
                    del output_data
        finally:
            pbar.update(1)
            pbar.set_postfix({"output data": output_data_logged})
    if len(output_data) > 0:
        _, new_shard = get_shards(output_dir)
        with open(new_shard, 'wb') as f:
            pickle.dump(output_data, f)
            all_output_data.extend(output_data)
            del output_data
    return all_output_data


def main():
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
    assert input_dir.exists(), input_dir

    project_dir = input_dir / args.project
    total = len(list((project_dir / 'raw_code').glob('*')))
    assert total > 0, 'No C files'
    logger.info(f'{total} items')

    output_dir = Path(args.output / args.project)
    output_dir.mkdir(exist_ok=True)

    preprocessed_dir = output_dir / 'preprocessed'
    preprocessed_dir.mkdir(exist_ok=True)
    preprocessed_data = preprocess(total, project_dir, preprocessed_dir, args.word2vec_name,
                                   args.shard_len)
    ggnn_input_dir = output_dir / 'ggnn_input'
    ggnn_input_dir.mkdir(exist_ok=True)
    split_and_save(preprocessed_data, ggnn_input_dir)


if __name__ == '__main__':
    main()
