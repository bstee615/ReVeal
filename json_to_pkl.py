import json
import pickle
from pathlib import Path

output_dir = Path('D:\\weile-lab\\thesis\\augmentation\\ReVeal\\out')

for n in ['train', 'valid', 'test']:
    json_f_name = output_dir / (n + '_GGNNinput.json')
    print(f'Reading from {json_f_name}')
    with open(json_f_name) as f:
        data = json.load(f)
    f_name = f_name.with_suffix('.pkl')
    print(f'Saving to {f_name}')
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)
    json_f_name.unlink()
