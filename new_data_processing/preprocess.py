import json
import logging
from pathlib import Path

import tqdm

from new_data_processing.extract_graph import get_graph
from new_data_processing.extract_slices import get_slices

logger = logging.getLogger(__name__)

import clang_stuff

from utils import read_csv


def preprocess(input_data, input_dir, output_dir, project, store=False, vuld_syse=False):
    if isinstance(input_data, Path):
        input_data = json.load(open(input_data))

    code_dir = input_dir / 'raw_code'
    parsed_dir = input_dir / 'parsed'
    assert code_dir.exists(), code_dir
    assert parsed_dir.exists(), parsed_dir
    logger.info(f'Number of Input Files: {len(input_data)}')

    if vuld_syse:
        clang_stuff.setup()

    output_data = []
    for i, d in tqdm.tqdm(enumerate(input_data)):
        file_name = d["file_name"]
        label = d["label"]
        code = d["code"]

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
        code_text = ' '.join(code.splitlines(keepends=True))
        t_code = clang_stuff.tokenize(code_text)
        if t_code is None:
            logger.debug(f'Skipping node {i} ({file_name}) because t_code is None')
            continue
        data_instance["tokenized"] = t_code

        # Get slices for VulDeePecker/SySeVR
        if vuld_syse:
            vuld_syse_slice_data = get_slices(combined_graph, nodes)
            data_instance.update(vuld_syse_slice_data)

        # Return data instance
        output_data.append(data_instance)
    logger.info(len(output_data), 'items')

    if store:
        with open(output_dir / (project + '_full_data_with_slices.json'), 'w') as output_file:
            json.dump(output_data, output_file)
        logger.info(f'saved output File to {output_file}')
    return output_data
