from chemgrams import *
from chemgrams import SMILESTokenizer

if __name__ == "__main__":

    # perplexity: 8.442949746533577
    # vocab = get_arpa_vocab('../resources/chemts_250k_smiles_klm_10gram_200429.arpa')
    # lm = KenLMSMILESLanguageModel('../resources/chemts_250k_smiles_klm_10gram_200429.klm', vocab)

    # 9.429387210856273
    vocab = get_arpa_vocab('../resources/chemts_250k_smiles_klm_6gram_200429.arpa')
    lm = KenLMSMILESLanguageModel('../resources/chemts_250k_smiles_klm_6gram_200429.klm', vocab)

    with open("../resources/zinc12_enaminebb_smiles_corpus.txt") as f:
        all_smiles = [s.strip() for s in f.readlines()]

    # the sum of log10 probs of each sentence in the corpus
    sum_log_prob = 0.0

    # the total number of "words" (i.e. tokens) in the corpus
    M = 0

    for smiles in all_smiles:
        tok = SMILESTokenizer(smiles.strip())
        tokens = tok.get_tokens()
        M += len(tokens)
        sum_log_prob += lm.log_prob(' '.join([t.value for t in tokens]))

    perplexity = 10**(-sum_log_prob/M)  # log probs are in base 10

    print(perplexity)
