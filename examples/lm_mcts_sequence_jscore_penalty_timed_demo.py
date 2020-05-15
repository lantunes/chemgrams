import os
import time
from threading import Timer
from chemgrams import *
from chemgrams.jscorer import JScorer
from chemgrams.logger import get_logger, log_top_best
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class StopTreeSearch(Exception):
    pass

if __name__ == '__main__':

    logger.info(os.path.basename(__file__))
    logger.info("KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)")
    logger.info("width = 24, max_depth = 100, start_state = ['<s>'], c = 5")
    logger.info("score: -1.0 if invalid; -1.0 if seen previously; J/(1+abs(J)) if valid")
    logger.info("LanguageModelMCTSWithPUCTTerminating")
    logger.info("JScorer")

    TIME_LIMIT = 8 * 60 * 60  # eight hours in seconds
    # TIME_LIMIT = 2*60  # 2 minutes in seconds

    LOG_INTERVAL = 2 * 60 * 60  # two hours in seconds
    # LOG_INTERVAL = 30.0  # 30 seconds

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

    num_simulations = 15000000  # much more than 8 hours
    width = 24
    max_depth = 100
    start_state = ["<s>"]
    c = 5

    sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
    logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
    cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
    jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

    all_smiles = {}
    num_valid = 0
    i = 0


    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num valid: %d" % num_valid)
        logger.info("num unique: %s" % len(all_smiles))
        logger.info("num iterations: %s" % i)
        log_top_best(all_smiles, 5, logger)
        t = Timer(LOG_INTERVAL, log_progress)
        t.start()
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()

    start = time.time()
    elapsed = time.time() - start

    def eval_function(text):
        global i, num_valid, all_smiles, elapsed

        if elapsed >= TIME_LIMIT:
            raise StopTreeSearch()

        i += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            elapsed = time.time() - start
            return -1.0

        num_valid += 1

        if smiles in all_smiles:
            score = -1.0
        else:
            jscore = jscorer.score(smiles)
            score = jscore / (1 + np.abs(jscore))
            all_smiles[smiles] = (jscore, generated)

        logger.debug("%s, %s" % (smiles, str(score)))
        elapsed = time.time() - start
        return score

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

    best = mcts.get_best_sequence()
    generated_text = ''.join(best[0])
    logger.info("best generated text: %s" % generated_text)
    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (smiles, jscorer.score(smiles), str((end - start))))

    log_top_best(all_smiles, 5, logger)
