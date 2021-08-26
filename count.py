# Count the number of samples in the after-GGNN dataset.
# Run from the folder after_ggnn containing *_GGNNinput_graph.jsons.
import json
import os
import tqdm
total = 0
target_count = [0, 0]
part_len = {}
part_count = {}
pbar = tqdm.tqdm(['train', 'valid', 'test'])
for part in pbar:
    part_target_count = [0, 0]
    if os.path.exists(part + '_GGNNinput_graph.json'):
        fname = part + '_GGNNinput_graph.json'
    elif os.path.exists(part + '_GGNNinput.json'):
        fname = part + '_GGNNinput.json'
    else:
        raise Exception('No file found')
    with open(fname) as f:
        pbar.set_description(f'Reading {fname}...')
        data = json.load(f)
        pbar.set_description(f'Counting {fname}...')
        for d in data:
            if "target" in d:
                target = d["target"]
            elif "targets" in d:
                target = d["targets"][0][0]
            target_count[target] += 1
            part_target_count[target] += 1
        part_len[part] = len(data)
        part_count[part] = part_target_count
        total += len(data)
print('part,len,non-buggy,buggy')
print('total', total, target_count[0], target_count[1], sep=',')
for part in ['train', 'valid', 'test']:
    print(part, part_len[part], part_count[part][0], part_count[part][1], sep=',')
