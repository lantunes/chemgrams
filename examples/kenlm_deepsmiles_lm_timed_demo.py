import os
import time
from threading import Timer

from chemgrams.logger import get_logger, log_top_best

from chemgrams import *
from chemgrams.jscorer import JScorer
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel(n=10, 'chemts_250k_deepsmiles_klm_10gram_200429.klm')")
logger.info("num_chars=100, text_seed='<s>'")
logger.info("JScorer")

TIME_LIMIT = 8*60*60  # eight hours in seconds
# TIME_LIMIT = 2*60  # 2 minutes in seconds

LOG_INTERVAL = 2*60*60  # two hours in seconds
# LOG_INTERVAL = 30.0  # 30 seconds

vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

all_smiles = {}
num_valid = 0
num_iterations = 0


def log_progress():
    global t
    logger.info("--results--")
    logger.info("num valid: %d" % num_valid)
    logger.info("num unique: %s" % len(all_smiles))
    logger.info("num iterations: %s" % num_iterations)
    log_top_best(all_smiles, 5, logger)
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()
t = Timer(LOG_INTERVAL, log_progress)
t.start()

logger.info("generating...")

start = time.time()
elapsed = time.time() - start
while elapsed < TIME_LIMIT:
    num_iterations += 1
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        num_valid += 1

        jscore = jscorer.score(sanitized)

        logger.debug("successful: %s , score: %s" % (sanitized, str(jscore)))

        all_smiles[sanitized] = (jscore, generated)

        if current_best_score is None or beats_current(jscore):
            current_best_score = jscore
            current_best_smiles = sanitized

    except Exception as e:
        pass

    elapsed = time.time() - start

end = time.time()

t.cancel()

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("num unique: %s" % len(all_smiles))
logger.info("num iterations: %s" % num_iterations)
logger.info("best: %s , score: %s (%s seconds)" % (current_best_smiles, str(current_best_score), str((end - start))))

log_top_best(all_smiles, 5, logger)
