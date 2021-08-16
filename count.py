# Count the number of samples in the after-GGNN dataset.
# Run from the folder after_ggnn containing *_GGNNinput_graph.jsons.
import json
total = 0
tc = [0, 0]
pl = {}
for part in ['train', 'valid', 'test']:
    with open(part + '_GGNNinput_graph.json') as f:
        data = json.load(f)
        for d in data:
            tc[d["target"]] += 1
        pl[part] = len(data)
        total += len(data)
print('bug:', tc[1], 'non-bug:', tc[0])
for part in ['train', 'valid', 'test']:
    print(part, pl[part], f'({pl[part]/total*100:.2f}%)')
