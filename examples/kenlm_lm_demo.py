import os
import time

from rdkit.RDLogger import logger

from chemgrams import *
from chemgrams.jscorer import JScorer
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

logger = logger()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


vocab = get_arpa_vocab('../models/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
lm = KenLMDeepSMILESLanguageModel('../models/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)

sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

start = time.time()
for i in range(100000):
    generated = lm.generate(num_chars=100, text_seed='<s>')
    try:
        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        jscore = jscorer.score(sanitized)

        logger.info("successful: %s , score: %s" % (sanitized, str(jscore)))

        if current_best_score is None or beats_current(jscore):
            current_best_score = jscore
            current_best_smiles = sanitized

    except Exception as e:
        pass
end = time.time()

logger.info("best: %s , score: %s (%s seconds)" % (current_best_smiles, str(current_best_score), str((end - start))))
