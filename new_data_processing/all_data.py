import argparse
import json

from tqdm import tqdm


def raw_code2dict(file_path):
    file_name = file_path.split('/')[-1]
    output = {
        'file_name': file_name,
        'label': int(file_name[-3]) if '_refactored' not in file_name else int(file_name[-(3 + len('_refactored'))]),
        'code': open(file_path, 'r').read()
    }
    return output


def create_ggnn_input(args):
    """
    Collate JSON file of all
    """
    args.input = Path(args.input)
    args.output = Path(args.output)

    raw_code = args.input / 'raw_code'
    output_data = []
    for cfile in tqdm(raw_code.glob('*')):
        fp = raw_code / cfile
        output_data.append(raw_code2dict(fp))

    output_file = args.output / (args.project + '_cfg_full_text_files.json')
    if args.store:
        with open(output_file, 'w') as of:
            json.dump(output_data, of)
            of.close()
        print(f'Saved Output File to {output_file}')


from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files',
                        choices=['chrome_debian', 'devign'], required=True)
    parser.add_argument('--input', help='input directory, containing <name>/{raw_code,parsed}', required=True)
    parser.add_argument('--output', help='output and intermediate processing directory', required=True)
    parser.add_argument('--store', help='store intermediate files?', action='store_true')
    args = parser.parse_args()

    create_ggnn_input(args)
