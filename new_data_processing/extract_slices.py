import json
import logging
from pathlib import Path

import tqdm
from graphviz import Digraph

logger = logging.getLogger(__name__)

from constants import l_funcs

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


def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)


def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph


def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass


def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)


def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1


def get_slices(combined_graph, nodes):
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()

    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name is None or function_name.strip() == '':
                continue
            if function_name.strip() in l_funcs:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)
    array_slices = []
    array_slices_bdir = []
    call_slices = []
    call_slices_bdir = []
    arith_slices = []
    arith_slices_bdir = []
    ptr_slices = []
    ptr_slices_bdir = []
    all_slices = []
    all_keys = set()
    _keys = set()
    for slice_ln in call_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        key = ' '.join([str(i) for i in all_slice_lines])
        if key not in _keys:
            call_slices.append(backward_sliced_lines)
            call_slices_bdir.append(all_slice_lines)
            _keys.add(key)
        if key not in all_keys:
            all_slices.append(all_slice_lines)
            all_keys.add(key)
    _keys = set()
    for slice_ln in array_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        key = ' '.join([str(i) for i in all_slice_lines])
        if key not in _keys:
            array_slices.append(backward_sliced_lines)
            array_slices_bdir.append(all_slice_lines)
            _keys.add(key)
        if key not in all_keys:
            all_slices.append(all_slice_lines)
            all_keys.add(key)
    _keys = set()
    for slice_ln in arithmatic_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        key = ' '.join([str(i) for i in all_slice_lines])
        if key not in _keys:
            arith_slices.append(backward_sliced_lines)
            arith_slices_bdir.append(all_slice_lines)
            _keys.add(key)
        if key not in all_keys:
            all_slices.append(all_slice_lines)
            all_keys.add(key)
    _keys = set()
    for slice_ln in ptr_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        key = ' '.join([str(i) for i in all_slice_lines])
        if key not in _keys:
            ptr_slices.append(backward_sliced_lines)
            ptr_slices_bdir.append(all_slice_lines)
            _keys.add(key)
        if key not in all_keys:
            all_slices.append(all_slice_lines)
            all_keys.add(key)
    return {
        'call_slices_vd': call_slices,
        'call_slices_sy': call_slices_bdir,
        'array_slices_vd': array_slices,
        'array_slices_sy': array_slices_bdir,
        'arith_slices_vd': arith_slices,
        'arith_slices_sy': arith_slices_bdir,
        'ptr_slices_vd': ptr_slices,
        'ptr_slices_sy': ptr_slices_bdir,
    }


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
