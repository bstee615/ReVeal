import itertools
import logging
import pickle
import re

logger = logging.getLogger(__name__)


def raw_code2dict(file_path):
    file_name = file_path.name
    # with open(file_path, encoding='utf-8', errors='ignore') as f:
    #     code = f.read()
    output = {
        'file_name': file_name,
        'label': int(file_name[-3]),
        # 'code': code
    }
    return output


def get_input_files(project_dir):
    raw_code = project_dir / 'raw_code'
    assert raw_code.exists(), f'{raw_code} does not exist'
    cfiles = list(raw_code.glob('*'))
    logger.info(f'got {len(cfiles)} files. Sorting...')
    cfiles = sorted(cfiles, key=lambda f: int(re.sub(r'[^0-9]', '', f.name)))
    logger.info(f'done sorting.')
    return cfiles


def read_input(cfiles, read_code=False):
    for i, cfile in enumerate(cfiles):
        file_name = cfile.name
        output = {
            'file_name': file_name,
            'label': int(file_name[-3]),
        }
        output["idx"] = i
        if read_code:
            with open(cfile, encoding='utf-8', errors='ignore') as f:
                output["code"] = f.read()
        yield output
