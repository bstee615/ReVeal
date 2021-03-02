import argparse
import csv
import json
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm


puncs = '~`!@#$%^&*()-+={[]}|\\;:\'\"<,>.?/'
puncs = list(puncs)


import re
import nltk
import warnings

warnings.filterwarnings('ignore')


type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

type_one_hot = np.eye(len(type_map))
# We currently consider 12 types of edges mentioned in ICST paper
edgeType_full = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}

# We currently consider 12 types of edges mentioned in ICST paper
edgeType_control = {
    'FLOWS_TO': 3,  # Control Flow
    'CONTROLS': 7,  # Control Dependency edge
}

edgeType_data = {
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
}

edgeType_control_data = {
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'FLOWS_TO': 3,  # Control Flow
    'CONTROLS': 7,  # Control Dependency edge
}


# edgeType = {'IS_AST_PARENT': 1}


#This is called by each of the graph building methods in turn
def inputGeneration(nodeCSV, edgeCSV, target, wv, edge_type_map, cfg_only=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])

    with open(nodeCSV, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodes = {}
        node_idx = 0
        for idx, node in enumerate(nodes):
            cfgNode = node['isCFGNode'].strip()
            if not cfg_only and (cfgNode == '' or cfgNode == 'False'):
                continue
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            node_split = nltk.word_tokenize(node_content)
            nrp = np.zeros(100)
            for token in node_split:
                try:
                    embedding = wv.wv[token]
                except:
                    embedding = np.zeros(100)
                nrp = np.add(nrp, embedding)
            if len(node_split) > 0:
                fNrp = np.divide(nrp, len(node_split))
            else:
                fNrp = nrp
            node_feature = type_one_hot[type_map[node_type] - 1].tolist()
            node_feature.extend(fNrp.tolist())
            allNodes[nodeKey] = node_feature
            nodeMap[nodeKey] = node_idx
            node_idx += 1

        if node_idx == 0 or node_idx >= 500:
            return None

        all_nodes_with_edges = set()
        trueNodeMap = {}
        all_edges = []

        with open(edgeCSV, 'r') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                start, end, eType = e["start"], e["end"], e["type"]
                if eType != "IS_FILE_OF":
                    if not start in nodeMap or not end in nodeMap or not eType in edge_type_map:
                        continue
                    all_nodes_with_edges.add(start)
                    all_nodes_with_edges.add(end)
                    edge = [start, edge_type_map[eType], end]
                    all_edges.append(edge)
        if len(all_edges) == 0:
            return None
        for i, node in enumerate(all_nodes_with_edges):
            trueNodeMap[node] = i
            gInput["node_features"].append(allNodes[node])
        for edge in all_edges:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            gInput["graph"].append(e)
    return gInput



#def unify_slices(list_of_list_of_slices):
#    taken_slice = set()
#    unique_slice_lines = []
#    for list_of_slices in list_of_list_of_slices:
#        for slice in list_of_slices:
#            slice_id = str(slice)
#            if slice_id not in taken_slice:
#                unique_slice_lines.append(slice)
#                taken_slice.add(slice_id)
#    return unique_slice_lines
#    pass


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='../data/')
    parser.add_argument('--project', default='chrome_debian')
    parser.add_argument('--csv', help='normalized csv files to process', default='../data/chrome_debian/parsed/')
    parser.add_argument('--src', help='source c files to process', default='../data/chrome_debian/raw_code/')
    parser.add_argument('--wv', default='../data/chrome_debian/raw_code_deb_chro.100')
    parser.add_argument('--output', default='../data/full_experiment_real_data/chrome_debian/chrome_debian.json')

    args = parser.parse_args()

    json_file_path = args.input + args.project + '_full_data_with_slices.json'
    data = json.load(open(json_file_path))
    model = Word2Vec.load(args.wv)

    final_data = []
    v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present = 0, 0, 0, 0, 0, 0, 0
    data_shard = 1

    # TQDM for loop for data elements
    for didx, entry in enumerate(tqdm(data)):

        file_name = entry['file_path'].split('/')[-1]
        nodes_path = os.path.join(args.csv, file_name, 'nodes.csv')
        edges_path = os.path.join(args.csv, file_name, 'edges.csv')

        label = int(entry['label'])

        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            continue

        #linized_code = {}
        #for ln, code in enumerate(entry['code'].split('\n')):
        #    linized_code[ln + 1] = code

        vuld_slices = {}

        syse_slices = {}

        graph_input_full = inputGeneration(
            nodes_path, edges_path, label, model, edgeType_full, False)

        graph_input_control = inputGeneration(
            nodes_path, edges_path, label, model, edgeType_control, True)

        graph_input_data = inputGeneration(nodes_path, edges_path, label, model, edgeType_data, True)

        graph_input_cd = inputGeneration(
            nodes_path, edges_path, label, model, edgeType_control_data, True)

        #draper_code = entry['tokenized']
        draper_code = {}

        if graph_input_full is None:
            continue

        if label == 1:
            v += 1
        else:
            nv += 1

        if len(vuld_slices) > 0: vd_present += 1
        if len(syse_slices) > 0: syse_present += 1

        if graph_input_control is not None: cg_present += 1
        if graph_input_data is not None: dg_present += 1
        if graph_input_cd is not None: cdg_present += 1

        data_point = {
            'id': didx,
            'file_name': file_name, 'file_path': os.path.abspath(entry['file_path']),
            'code': entry['code'],
            'vuld': vuld_slices, 'vd_present': 1 if len(vuld_slices) > 0 else 0,
            'syse': syse_slices, 'syse_present': 1 if len(syse_slices) > 0 else 0,
            'draper': draper_code,
            'full_graph': graph_input_full,
            'cgraph': graph_input_control,
            'dgraph': graph_input_data,
            'cdgraph': graph_input_cd,
            'label': int(entry['label'])
        }

        final_data.append(data_point)
        if len(final_data) == 5000:
            output_path = args.output + '.shard' + str(data_shard)
            with open(output_path, 'w') as fp:
                json.dump(final_data, fp)
                fp.close()
            print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')
            final_data = []
            data_shard += 1
    print("Vulnerable:\t%d\n"
          "Non-Vul:\t%d\n"
          "VulDeePecker:\t%d\n"
          "SySeVr:\t%d\n"
          "Control: %d\tData: %d\tBoth: %d" % \
          (v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present))
    output_path = args.output + '.shard' + str(data_shard)
    with open(output_path, 'w') as fp:
        json.dump(final_data, fp)
        fp.close()
    print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')


if __name__ == '__main__':
    main()
