from chemgrams import *
from rdkit.RDLogger import logger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

logger = logger()


if __name__ == '__main__':

    logger.info("loading language model...")
    lm = DeepSMILESLanguageModelUtils.get_lm("../models/chembl_25_deepsmiles_nltklm_5gram_190330.pkl")

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

        score = len(tokenized.get_tokens()) / (text_length - 1)  # provide more reward for longer text sequences

        logger.info("%s, %s" % (generated, str(score)))
        return score

    mcts = LanguageModelMCTSWithUCB1(lm, width, text_length, eval_function)
    state = start_state

    logger.info("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    logger.info("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text)
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    logger.info("SMILES: %s" % smiles)
