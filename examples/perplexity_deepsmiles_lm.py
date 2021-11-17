from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel
from chemgrams import DeepSMILESTokenizer

import pybel
from deepsmiles import Converter

if __name__ == "__main__":

    # 3.747563829583387
    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)

    # 3.6372667959019758
    # vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
    # lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

    converter = Converter(rings=True, branches=True)


    def smiles_to_deepsmiles(smiles):
        canonical = pybel.readstring("smi", smiles).write("can").strip()
        return converter.encode(canonical)

    with open("../resources/zinc12_enaminebb_smiles_corpus.txt") as f:
        all_smiles = [s.strip() for s in f.readlines()]

    # the sum of log10 probs of each sentence in the corpus
    sum_log_prob = 0.0

    # the total number of "words" (i.e. tokens) in the corpus
    M = 0

    for smiles in all_smiles:
        dsmi = smiles_to_deepsmiles(smiles.strip())
        tok = DeepSMILESTokenizer(dsmi)
        tokens = tok.get_tokens()
        M += len(tokens)
        sum_log_prob += lm.log_prob(' '.join([t.value for t in tokens]))

    perplexity = 10**(-sum_log_prob/M)  # log probs are in base 10

    print(perplexity)
