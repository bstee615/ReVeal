import itertools
import logging
import pickle

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


def get_input(project_dir, start=0):
    raw_code = project_dir / 'raw_code'
    assert raw_code.exists(), raw_code
    cached_raw_code = raw_code / 'cache.pkl'
    if cached_raw_code.exists():
        with open(cached_raw_code, 'rb') as f:
            return iter(pickle.load(f))
    else:
        cfiles = raw_code.glob('*')
        cfiles = itertools.islice(cfiles, start, None)
        all_outputs = []
        for i, cfile in enumerate(cfiles):
            output = raw_code2dict(cfile)
            output["idx"] = i
            all_outputs.append(output)
            yield output
