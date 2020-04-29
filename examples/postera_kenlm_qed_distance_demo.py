import os

from rdkit import rdBase
from rdkit.Chem import QED

from chemgrams import *
from chemgrams.logger import get_logger
from chemgrams.tanimotoscorer import TanimotoScorer

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.klm')")
logger.info("num_chars=100, text_seed='<s>'")

vocab = get_arpa_vocab('../resources/postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_submissions_2020_03_25_deepsmiles_klm_6gram.klm', vocab)


def log_top_best_with_props(pairs, props, top_n, lgg):
    all_best = reversed(list(reversed(sorted(pairs.items(), key=lambda kv: kv[1][0])))[:top_n])
    for i, ab in enumerate(all_best):
        molprops = props[ab[0]]
        lgg.info("%d. %s, %s, QED: %s, mean similarity.: %s, max similarity.: %s, most similar: %s" %
                 (top_n - i, ab[0], str(ab[1]), molprops[0], molprops[1], molprops[2], molprops[3]))


logger.info("reading submissions...")
with open("../models/postera-covid-submissions-2020-03-25.txt") as f:
    submissions = f.readlines()
submissions = [Chem.MolFromSmiles(x.strip()) for x in submissions]
logger.info("finished reading %s submissions" % len(submissions))

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

mol_props = {}
all_smiles = {}
num_valid = 0

for i in range(250000):
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        num_valid += 1

        mol = Chem.MolFromSmiles(sanitized)
        qedscore = QED.qed(mol)
        distances = []
        most_similar_submission_mol = None
        max_distance = 0.0
        for submission in submissions:
            distance = TanimotoScorer.score_mols(submission, mol)
            distances.append(distance)
            if distance > max_distance:
                max_distance = distance
                most_similar_submission_mol = submission
        mean_distance = np.mean(distances)

        # 0.8*qedscore + 0.2*(1 - mean_distance) -> higher score for less similarity to submissions
        # 0.8*qedscore + 0.2*mean_distance -> higher score for more similarity to submissions
        tot_score = 0.8*qedscore + 0.2*(1 - mean_distance)

        all_smiles[sanitized] = (tot_score, generated)

        mol_props[sanitized] = (qedscore, mean_distance, max_distance, Chem.MolToSmiles(most_similar_submission_mol))

        if current_best_score is None or beats_current(tot_score):
            current_best_score = tot_score
            current_best_smiles = sanitized

    except Exception as e:
        pass

    if (i+1) % 5000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        log_top_best_with_props(all_smiles, mol_props, 5, logger)

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))

log_top_best_with_props(all_smiles, mol_props, 5000, logger)
