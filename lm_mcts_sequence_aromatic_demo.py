from chemgrams import *
from rdkit import Chem

if __name__ == '__main__':

    print("loading language model...")
    # lm = DeepSMILESLanguageModelUtils.get_lm("models/chembl_25_deepsmiles_lm_5gram_190330.pkl")

    vocab = get_arpa_vocab('models/chembl_25_deepsmiles_klm_6gram_190413.arpa')
    lm = DeepSMILESKenLM('models/chembl_25_deepsmiles_klm_6gram_190413.klm', vocab)

    num_simulations = 1000
    width = 3
    text_length = 25
    start_state = ["<s>"]

    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return 0
        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        num_aromatic_atoms = 0
        for i in range(num_atoms):
            if mol.GetAtomWithIdx(i).GetIsAromatic():
                num_aromatic_atoms += 1
        return num_aromatic_atoms / 23

    mcts = LanguageModelMCTS(lm, width, text_length, eval_function)
    state = start_state

    print("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    print("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    print("SMILES: %s" % smiles)
