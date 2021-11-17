import os
import time
from threading import Timer
import numpy as np

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.logger import get_logger, log_top_best

from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel(n=10, 'chembl_25_deepsmiles_klm_10gram_200503.klm')")
logger.info("num_chars=100, text_seed='<s>'")
logger.info("TanimotoScorer(abilify, radius=6)")

TIME_LIMIT = 3 * 60 * 60  # three hours in seconds
# TIME_LIMIT = 2*60  # 2 minutes in seconds

LOG_INTERVAL = 1 * 60 * 60  # one hour in seconds
# LOG_INTERVAL = 30.0  # 30 seconds

KEEP_TOP_N = 20000

vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
distance_scorer = TanimotoScorer(abilify, radius=6)

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

all_unique = {}
all_valid = []
num_valid = 0
num_iterations = 0


def log_progress():
    global t
    logger.info("--results--")
    logger.info("num valid: %d" % num_valid)
    logger.info("num unique: %s" % len(all_unique))
    logger.info("num iterations: %s" % num_iterations)
    log_top_best(all_unique, 5, logger)
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
        mol = Chem.MolFromSmiles(sanitized)
        if mol is None: raise Exception

        num_valid += 1

        score = distance_scorer.score_mol(mol)

        logger.debug("successful: %s , score: %s" % (sanitized, str(score)))

        all_unique[sanitized] = (score, generated)
        all_valid.append((sanitized, score))

        if current_best_score is None or beats_current(score):
            current_best_score = score
            current_best_smiles = sanitized

    except Exception as e:
        pass

    elapsed = time.time() - start

end = time.time()

t.cancel()

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("num unique: %s" % len(all_unique))
logger.info("num iterations: %s" % num_iterations)
logger.info("best: %s , score: %s (%s seconds)" % (current_best_smiles, str(current_best_score), str((end - start))))

log_top_best(all_unique, 5, logger)

all_valid_scores = []
for smi in list(reversed(sorted(all_valid, key=lambda i: i[1])))[:KEEP_TOP_N]:
    all_valid_scores.append(smi[1])
logger.info('all valid: size: %s, mean score: %s' % (len(all_valid_scores), np.mean(all_valid_scores)))
