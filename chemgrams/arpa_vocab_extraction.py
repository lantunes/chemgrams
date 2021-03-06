
def get_arpa_vocab(arpa_file_path, exclude_unk=True):
    with open(arpa_file_path, 'r') as f:
        lines = f.readlines()

    tokens = []

    in_1_grams = False
    for line in lines:
        if line.startswith('\\1'):
            in_1_grams = True
        if in_1_grams and not line.startswith('\\'):
            vals = line.split('\t')
            if len(vals) == 3:
                if exclude_unk and vals[1] == '<unk>':
                    continue
                tokens.append(vals[1])
        if in_1_grams and line.startswith("\\2"):
            break

    return tokens
