import csv

import nltk
import numpy as np

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

edgeType = {
    "full_graph": edgeType_full,
    "cgraph": edgeType_control,
    "dgraph": edgeType_data,
    "cdgraph": edgeType_control_data,
}


def get_ggnn_graph(nodeCSV, edgeCSV, target, wv, portion):
    edge_type_map = edgeType[portion]
    if portion == 'full_graph':
        cfg_only = False
    else:
        cfg_only = True

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
