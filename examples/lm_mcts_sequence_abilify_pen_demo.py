import os
import time
from chemgrams import *
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.logger import get_logger, log_top_best
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    logger.info(os.path.basename(__file__))
    logger.info("KenLMDeepSMILESLanguageModel('../models/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)")
    logger.info("width = 24, max_depth = 100, start_state = ['<s>'], c = 5")
    logger.info("score: -1.0 if invalid; -1.0 if seen previously; tanimoto distance from abilify if valid")
    logger.info("LanguageModelMCTSWithPUCTTerminating")
    logger.info("TanimotoScorer(abilify)")

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../models/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
    lm = KenLMDeepSMILESLanguageModel('../models/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)

    num_simulations = 500000
    width = 12
    max_depth = 35
    start_state = ["<s>"]
    c = 5

    abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
    scorer = TanimotoScorer(abilify)

    all_smiles = {}
    num_valid = 0
    i = 0

    def log_best(n, all_best, n_valid, lggr):
        if n % 1000 == 0:
            lggr.info("--iteration: %d--" % n)
            lggr.info("num valid: %d" % n_valid)
            log_top_best(all_best, 5, lggr)

    def eval_function(text):
        global i, num_valid, all_smiles
        i += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            log_best(i, all_smiles, num_valid, logger)
            return -1.0

        num_valid += 1

        if smiles in all_smiles:
            score = -1.0
        else:
            score = scorer.score(smiles)
            all_smiles[smiles] = (score, generated)

        logger.debug("%s, %s" % (smiles, str(score)))
        log_best(i, all_smiles, num_valid, logger)
        return score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")
    start = time.time()
    mcts.search(state, num_simulations)
    end = time.time()

    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)

    best = mcts.get_best_sequence()
    generated_text = ''.join(best[0])
    logger.info("best generated text: %s" % generated_text)
    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (smiles, scorer.score(smiles), str((end - start))))

    log_top_best(all_smiles, 5, logger)
