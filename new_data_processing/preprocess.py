import json
import logging
from pathlib import Path

import tqdm

from new_data_processing.extract_slices import get_slices

logger = logging.getLogger(__name__)

import clang_stuff

from utils import read_csv


def extract_nodes_with_location_info(nodes):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number
    pass


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS':  # Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES':  # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph


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

        # Make graph data structure
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
        combined_graph = combine_control_and_data_adjacents(adjacency_list)

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
