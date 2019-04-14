from chemgrams import *
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem

if __name__ == '__main__':

    print("loading language model...")
    lm = DeepSMILESLanguageModelUtils.get_lm("models/chembl_25_deepsmiles_nltklm_5gram_190330.pkl")

    num_simulations = 1000
    width = 3
    text_length = 25
    start_state = ["<M>"]

    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated)
            DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return 0
        extracted = DeepSMILESLanguageModelUtils.extract(generated)
        tokenized = DeepSMILESTokenizer(extracted)
        len_reward = len(tokenized.get_tokens()) / (text_length - 1)  # provide more reward for longer text sequences
        perplexity = lm.perplexity(text)
        perplexity_reward = perplexity / (1 + perplexity)
        return (perplexity_reward*0.5) + (len_reward*0.5)

    mcts = LanguageModelMCTSWithUCB1(lm, width, text_length, eval_function)
    state = start_state

    print("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    print("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text)
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    print("SMILES: %s" % smiles)
