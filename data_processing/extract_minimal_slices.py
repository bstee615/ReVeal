import os
import json
import csv
import numpy as np
import re 
import warnings
from tqdm import tqdm
import sys, argparse

warnings.filterwarnings('ignore')


def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
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


def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[ln + 1] = line
        return code_lines




    


def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        return ' '.join(lines)
    
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--project', help='name of project for differentiating files', default='chrome_debian')
    parser.add_argument('--input_raw', help='directory where raw code and parsed are stored', default='../data/chrome_debian/raw_code')
    #parser.add_argument('--input_parse', help='directory where raw code and parsed are stored', default='../data/chrome_debian/parsed_code')
    parser.add_argument('--text_in', help='path to cfg_full_text_files', default='../data/ggnn_input/')
    parser.add_argument('--output', help='output path for sliced full data json', default='../data/')

    args = parser.parse_args()

    split_dir = args.input_raw
    #parsed = args.input_parse
    #split_dir = args.input + 'raw_code/'
    #parsed = args.input + 'parsed/'

    ggnn_json_data = json.load(open(args.text_in + args.project + '_cfg_full_text_files.json'))

    files = [d['file_name'] for d in ggnn_json_data]
    print(f'Number of Input Files: {len(files)}')

    all_data = []
        
    for i, file_name  in enumerate(files):
        label = file_name.strip()[:-2].split('_')[-1]
        code_text = read_file(split_dir + file_name.strip())
 
                
#        if t_code is None:
#            continue

        data_instance = {
            'file_path': split_dir + file_name.strip(),
            'code' : code_text,
#            'tokenized': t_code,
#            'call_slices_vd': call_slices,
#            'call_slices_sy': call_slices_bdir,
#           'array_slices_vd': array_slices,
#            'array_slices_sy': array_slices_bdir,
#            'arith_slices_vd': arith_slices,
#            'arith_slices_sy': arith_slices_bdir,
#            'ptr_slices_vd': ptr_slices,
#            'ptr_slices_sy': ptr_slices_bdir,
            'label': int(label)
        }

        all_data.append(data_instance)
        
        #if i % 1000 == 0:
        #    print(i, len(call_slices), len(call_slices_bdir), 
        #        len(array_slices), len(array_slices_bdir), 
        #        len(arith_slices), len(arith_slices_bdir), sep='\t')


    print(len(all_data))
    print('Writing output data to: ',(args.output + args.project + '_full_data_with_slices.json'))

    output_file = open(args.output + args.project + '_full_data_with_slices.json', 'w')
    json.dump(all_data, output_file)

    output_file.close()
