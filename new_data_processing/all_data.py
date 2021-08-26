import argparse
import logging
from pathlib import Path

from new_data_processing.create_ggnn_input import create_ggnn_input
from new_data_processing.extract_slices import extract_slices

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files',
                        choices=['chrome_debian', 'devign'], required=True)
    parser.add_argument('--input', help='input directory, containing <name>/{raw_code,parsed}', required=True)
    parser.add_argument('--output', help='output and intermediate processing directory', required=True)
    parser.add_argument('--store', help='store intermediate files?', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    project = args.project
    code_dir = input_dir / project / 'raw_code'
    parsed_dir = input_dir / project / 'parsed'

    full_text_files = create_ggnn_input(input_dir, output_dir, project)
    full_text_files_with_slices = extract_slices(full_text_files, input_dir, output_dir, project)
    ggnn_data = create_ggnn_data(args, full_text_files)
