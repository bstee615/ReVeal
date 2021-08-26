import argparse
from pathlib import Path

from new_data_processing.create_ggnn_input import create_ggnn_input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files',
                        choices=['chrome_debian', 'devign'], required=True)
    parser.add_argument('--input', help='input directory, containing <name>/{raw_code,parsed}', required=True)
    parser.add_argument('--output', help='output and intermediate processing directory', required=True)
    parser.add_argument('--store', help='store intermediate files?', action='store_true')
    args = parser.parse_args()

    args.input = Path(args.input)
    args.output = Path(args.output)

    create_ggnn_input(args)
