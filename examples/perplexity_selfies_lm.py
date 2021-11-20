from chemgrams import get_arpa_vocab, KenLMSELFIESLanguageModel

import selfies as sf
from openbabel import pybel


if __name__ == "__main__":

    # 6.655716427238491
    vocab = get_arpa_vocab('../resources/chemts_250k_selfies_klm_10gram_210908.arpa')
    lm = KenLMSELFIESLanguageModel('../resources/chemts_250k_selfies_klm_10gram_210908.klm', vocab)

    def smiles_to_selfies(smiles):
        canonical = pybel.readstring("smi", smiles).write("can").strip()
        return sf.encoder(canonical)

    with open("../resources/zinc12_enaminebb_smiles_corpus.txt") as f:
        all_smiles = [s.strip() for s in f.readlines()]

    # the sum of log10 probs of each sentence in the corpus
    sum_log_prob = 0.0

    # the total number of "words" (i.e. tokens) in the corpus
    M = 0

    for smiles in all_smiles:
        s = smiles_to_selfies(smiles.strip())
        tokens = list(sf.split_selfies(s))
        M += len(tokens)
        sum_log_prob += lm.log_prob(' '.join(tokens))

    perplexity = 10**(-sum_log_prob/M)  # log probs are in base 10

    print(perplexity)
