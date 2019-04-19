from chemgrams import *
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem
from rdkit.RDLogger import logger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()


if __name__ == '__main__':

    logger.info("loading language model...")
    lm = DeepSMILESLanguageModelUtils.get_lm("../models/chembl_25_deepsmiles_nltklm_5gram_190330.pkl")

    num_simulations = 1000
    width = 6
    text_length = 25
    start_state = ["<M>"]

    # # maximizing logP
    # logp_max = 10.143
    # logp_min = -3.401
    # factor = 1

    # minimizing logP
    logp_min = -10.143
    logp_max = 3.401
    factor = -1


    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated)
            DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return 0
        # extracted = DeepSMILESLanguageModelUtils.extract(generated)
        # tokenized = DeepSMILESTokenizer(extracted)
        # len_reward = len(tokenized.get_tokens()) / (text_length - 1)  # provide more reward for longer text sequences

        perplexity = lm.perplexity(text)
        perplexity_reward = perplexity / (1 + perplexity)

        decoded = DeepSMILESLanguageModelUtils.decode(generated)
        smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        mol = Chem.MolFromSmiles(smiles)
        logp = factor * MolLogP(mol)
        logp_score = (logp - logp_min)/(logp_max - logp_min)  # normalize logP between 0 and 1

        score = (logp_score * 0.5) + (perplexity_reward * 0.5)

        logger.info("%s, %s" % (generated, str(score)))
        return score

    # mcts = LanguageModelMCTSWithUCB1(lm, width, text_length, eval_function)
    mcts = LanguageModelMCTSWithPUCT(lm, width, text_length, eval_function, cpuct=5)
    state = start_state

    logger.info("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    logger.info("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text)
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    mol = Chem.MolFromSmiles(smiles)
    logp = MolLogP(mol)
    logger.info("SMILES: %s, logP: %s" % (smiles, logp))
