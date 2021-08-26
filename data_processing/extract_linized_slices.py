import re

import nltk

from new_data_processing.constants import keywords, puncs, l_funcs


def symbolic_tokenize(code):
    tokens = nltk.word_tokenize(code)
    c_tokens = []
    for t in tokens:
        if t.strip() != '':
            c_tokens.append(t.strip())
    f_count = 1
    var_count = 1
    symbol_table = {}
    final_tokens = []
    for idx in range(len(c_tokens)):
        t = c_tokens[idx]
        if t in keywords:
            final_tokens.append(t)
        elif t in puncs:
            final_tokens.append(t)
        elif t in l_funcs:
            final_tokens.append(t)
        elif (idx + 1) < len(c_tokens) and c_tokens[idx + 1] == '(':
            if t in keywords:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t])
            idx += 1

        elif t.endswith('('):
            t = t[:-1]
            if t in keywords:
                final_tokens.append(t + '(')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '(')
        elif t.endswith('()'):
            t = t[:-2]
            if t in keywords:
                final_tokens.append(t + '( )')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '( )')
        elif re.match("^\"*\"$", t) is not None:
            final_tokens.append("STRING")
        elif re.match("^[0-9]+(\.[0-9]+)?$", t) is not None:
            final_tokens.append("NUMBER")
        elif re.match("^[0-9]*(\.[0-9]+)$", t) is not None:
            final_tokens.append("NUMBER")
        else:
            if t not in symbol_table.keys():
                symbol_table[t] = "VAR" + str(var_count)
                var_count += 1
            final_tokens.append(symbol_table[t])
    return ' '.join(final_tokens)


def extract_slices(linized_code, list_of_slices):
    sliced_codes = []
    for slice in list_of_slices:
        tokenized = []
        for ln in slice:
            code = linized_code[ln]
            tokenized.append(symbolic_tokenize(code))
        sliced_codes.append(' '.join(tokenized))
    return sliced_codes
    pass


def unify_slices(list_of_list_of_slices):
    taken_slice = set()
    unique_slice_lines = []
    for list_of_slices in list_of_list_of_slices:
        for slice in list_of_slices:
            slice_id = str(slice)
            if slice_id not in taken_slice:
                unique_slice_lines.append(slice)
                taken_slice.add(slice_id)
    return unique_slice_lines
    pass


def get_linized_slices(code, entry):
    linized_code = {}
    for ln, code in enumerate(code.split('\n')):
        linized_code[ln + 1] = code
    vuld_slices = extract_slices(linized_code, entry['call_slices_vd'])
    syse_slices = extract_slices(
        linized_code, unify_slices(
            [entry['call_slices_sy'], entry['array_slices_sy'], entry['arith_slices_sy'], entry['ptr_slices_sy']]
        )
    )
    return {
        'vuld': vuld_slices,
        'vd_present': 1 if len(vuld_slices) > 0 else 0,
        'syse': syse_slices,
        'syse_present': 1 if len(syse_slices) > 0 else 0,
    }
