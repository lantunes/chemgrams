import os
import time
from threading import Timer
import numpy as np

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, StopTreeSearch, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.logger import get_logger, log_top_best

from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    logger.info(os.path.basename(__file__))
    logger.info("KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)")
    logger.info("width = 12, max_depth = 50, start_state = ['<s>'], c = 5")
    logger.info("score: -1.0 if invalid; -1.0 if seen previously; TanimotoScorer(abilify, radius=6) if valid; rescaling from [0,1] to [-1,1]")
    logger.info("LanguageModelMCTSWithPUCTTerminating")

    TIME_LIMIT = 3 * 60 * 60  # three hours in seconds
    # TIME_LIMIT = 2*60  # 2 minutes in seconds

    LOG_INTERVAL = 1 * 60 * 60  # one hour in seconds
    # LOG_INTERVAL = 30.0  # 30 seconds

    KEEP_TOP_N = 20000

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

    abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
    distance_scorer = TanimotoScorer(abilify, radius=6)

    num_simulations = 15000000  # much more than 8 hours
    width = 12
    max_depth = 50
    start_state = ["<s>"]
    c = 5

    all_unique = {}
    all_valid = []
    num_valid = 0
    simulations = 0

    current_best_score = None
    current_best_smiles = None
    beats_current = lambda sc: sc > current_best_score


    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num valid: %d" % num_valid)
        logger.info("num unique: %s" % len(all_unique))
        logger.info("num iterations: %s" % simulations)
        log_top_best(all_unique, 5, logger)
        t = Timer(LOG_INTERVAL, log_progress)
        t.start()
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()

    start = time.time()
    elapsed = time.time() - start

    def eval_function(text):
        global simulations, num_valid, all_unique, all_valid, elapsed, current_best_score, current_best_smiles, beats_current

        if elapsed >= TIME_LIMIT:
            raise StopTreeSearch()

        simulations += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: raise Exception
        except Exception:
            elapsed = time.time() - start
            return -1.0

        num_valid += 1
        score = distance_scorer.score_mol(mol)
        all_unique[smiles] = (score, generated)

        if current_best_score is None or beats_current(score):
            current_best_score = score
            current_best_smiles = smiles

        all_valid.append((smiles, score))

        ret_score = -1.0 if smiles in all_unique else score

        # rescale score from [0,1] to [-1,1]
        ret_score = (ret_score * 2) + (-1) if ret_score >= 0. else ret_score

        elapsed = time.time() - start
        return ret_score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")

    try:
        mcts.search(state, num_simulations)
    except StopTreeSearch:
        pass

    t.cancel()
    end = time.time()

    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)
    logger.info("num unique: %s" % len(all_unique))
    logger.info("num iterations: %s" % simulations)
    logger.info("best: %s, score: %s (%s seconds)" % (current_best_smiles, current_best_score, str((end - start))))

    log_top_best(all_unique, 5, logger)

    all_valid_scores = []
    for smi in list(reversed(sorted(all_valid, key=lambda i: i[1])))[:KEEP_TOP_N]:
        all_valid_scores.append(smi[1])
    logger.info('all valid: size: %s, mean score: %s' % (len(all_valid_scores), np.mean(all_valid_scores)))
