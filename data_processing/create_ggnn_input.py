import logging

logger = logging.getLogger(__name__)


def raw_code2dict(file_path):
    file_name = file_path.name
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        code = f.read()
    output = {
        'file_name': file_name,
        'label': int(file_name[-3]),
        'code': code
    }
    return output


def get_input(project_dir):
    raw_code = project_dir / 'raw_code'
    cfiles = raw_code.glob('*')
    for i, cfile in enumerate(cfiles):
        output = raw_code2dict(cfile)
        output["idx"] = i
        yield output
