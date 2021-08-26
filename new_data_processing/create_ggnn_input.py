import os
import json
import argparse
from tqdm import tqdm

def raw_code2dict(file_path):
    file_name = file_path.split('/')[-1]
    output = {
        'file_name':file_name,
        'label':int(file_name[-3]) if '_refactored' not in file_name else int(file_name[-(3+len('_refactored'))]),
        'code':open(file_path, 'r').read()
        }
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files', default='chrome_debian')
    parser.add_argument('--input', help='directory where raw code and parsed are stored', default='../data/chrome_debian')
    parser.add_argument('--output', help='output directory for resulting json file', default='../data/ggnn_input/')
    args = parser.parse_args()

    code_file_path = args.input + '/raw_code/'

    output_data = []
    for cfile in tqdm(os.listdir(code_file_path)):
        fp = code_file_path + cfile
        output_data.append(raw_code2dict(fp))
    
    output_file = args.output + args.project + '_cfg_full_text_files.json'

    with open(output_file, 'w') as of:
        json.dump(output_data, of)
        of.close()

    print(f'Saved Output File to {output_file}')
