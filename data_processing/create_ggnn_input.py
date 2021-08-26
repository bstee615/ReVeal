import json
import logging

import tqdm

logger = logging.getLogger(__name__)


def raw_code2dict(file_path):
    file_name = file_path.split('/')[-1]
    output = {
        'file_name': file_name,
        'label': int(file_name[-3]) if '_refactored' not in file_name else int(file_name[-(3 + len('_refactored'))]),
        'code': open(file_path, 'r').read()
    }
    return output


def create_ggnn_input(code_dir, output_dir, project, store=False):
    """
    Collate JSON file of all
    """

    raw_code = code_dir / 'raw_code'
    cfiles = raw_code.glob('*')
    total = len(list(raw_code.glob('*')))
    logger.info(f'{len(cfiles)} items')

    output_data = []
    for i, cfile in tqdm.tqdm(enumerate(cfiles)):
        fp = raw_code / cfile
        output = raw_code2dict(fp)
        output["idx"] = i
        output_data.append(output)

    output_file = output_dir / (project + '_cfg_full_text_files.json')
    if store:
        with open(output_file, 'w') as of:
            json.dump(output_data, of)
            of.close()
        logger.info(f'saved output File to {output_file}')
    return total, output_data
