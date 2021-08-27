def read_csv(csv_file_path):
    data = []
    with open(csv_file_path, encoding='utf-8', errors='ignore') as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def get_shards(output_dir):
    old_shard_filenames = []
    shard_idx = 0
    shard_filename = output_dir / f'preprocessed_shard{shard_idx}.pkl'
    while shard_filename.exists():
        old_shard_filenames.append(shard_filename)
        shard_idx += 1
        shard_filename = output_dir / f'preprocessed_shard{shard_idx}.pkl'
    return old_shard_filenames, shard_filename
