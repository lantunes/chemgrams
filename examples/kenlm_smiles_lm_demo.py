import os
import time
import numpy as np

from chemgrams import get_arpa_vocab, KenLMSMILESLanguageModel, SMILESLanguageModelUtils
from chemgrams.jscorer import JScorer
from chemgrams.logger import get_logger, log_top_best

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMSMILESLanguageModel(n=10, 'chemts_250k_smiles_klm_10gram_200429.klm')")
logger.info("num_chars=100, text_seed='<s>'")
logger.info("JScorer")


vocab = get_arpa_vocab('../resources/chemts_250k_smiles_klm_10gram_200429.arpa')
lm = KenLMSMILESLanguageModel('../resources/chemts_250k_smiles_klm_10gram_200429.klm', vocab)

sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

all_smiles = {}
num_valid = 0

start = time.time()
for i in range(3000000):
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        extracted = SMILESLanguageModelUtils.extract(generated, start='<s>', end='</s>')
        sanitized = SMILESLanguageModelUtils.sanitize(extracted)

        num_valid += 1

        jscore = jscorer.score(sanitized)

        logger.debug("successful: %s , score: %s" % (sanitized, str(jscore)))

        all_smiles[sanitized] = (jscore, generated)

        if current_best_score is None or beats_current(jscore):
            current_best_score = jscore
            current_best_smiles = sanitized

    except Exception as e:
        pass

    if (i+1) % 50000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        logger.info("num unique: %s" % len(all_smiles))
        log_top_best(all_smiles, 5, logger)

end = time.time()

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("num unique: %s" % len(all_smiles))
logger.info("best: %s , score: %s (%s seconds)" % (current_best_smiles, str(current_best_score), str((end - start))))

log_top_best(all_smiles, 5, logger)
