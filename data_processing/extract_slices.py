from new_data_processing.constants import l_funcs


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
