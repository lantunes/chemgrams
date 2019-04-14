from chemgrams import *
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
        decoded = DeepSMILESLanguageModelUtils.decode(generated)
        smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        num_aromatic_atoms = 0
        for i in range(num_atoms):
            if mol.GetAtomWithIdx(i).GetIsAromatic():
                num_aromatic_atoms += 1
        arom_reward = num_aromatic_atoms / 23

        perplexity = lm.perplexity(text)
        perplexity_reward = perplexity / (1 + perplexity)

        return (perplexity_reward * 0.5) + (arom_reward * 0.5)

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
