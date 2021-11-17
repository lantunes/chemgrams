from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithUCB1

from rdkit.RDLogger import logger
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()


if __name__ == '__main__':

    logger.info("loading language model...")
    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

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
        arom_reward = num_aromatic_atoms / 23

        perplexity = lm.perplexity(text)
        perplexity_reward = perplexity / (1 + perplexity)

        score = (perplexity_reward * 0.5) + (arom_reward * 0.5)

        logger.info("%s, %s" % (generated, str(score)))
        return score

    mcts = LanguageModelMCTSWithUCB1(lm, width, text_length, eval_function)
    state = start_state

    logger.info("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    logger.info("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    logger.info("SMILES: %s" % smiles)
