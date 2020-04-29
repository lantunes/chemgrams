import os

from rdkit import rdBase
from rdkit.Chem import QED

from chemgrams import *
from chemgrams.logger import get_logger, log_top_best
import chemgrams.sascorer as sascorer

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.klm')")
logger.info("num_chars=100, text_seed='<s>'")

vocab = get_arpa_vocab('../resources/postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.klm', vocab)

sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
sa_mean = np.mean(sa_scores)
sa_std = np.std(sa_scores)
current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

all_smiles = {}
num_valid = 0

for i in range(25000):
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        num_valid += 1

        mol = Chem.MolFromSmiles(sanitized)
        qedscore = QED.qed(mol)
        sa_score = -sascorer.calculateScore(mol)
        sa_score_norm = (sa_score - sa_mean) / sa_std
        tot_score = 0.9*qedscore + 0.1*sa_score_norm

        all_smiles[sanitized] = (tot_score, generated)

        if current_best_score is None or beats_current(tot_score):
            current_best_score = tot_score
            current_best_smiles = sanitized

    except Exception as e:
        pass

    if (i+1) % 5000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        log_top_best(all_smiles, 5, logger)

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))

log_top_best(all_smiles, 5, logger)
