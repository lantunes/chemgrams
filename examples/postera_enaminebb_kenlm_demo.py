import os

from rdkit import rdBase
from rdkit.Chem import QED
from rdkit.Chem import AllChem

from chemgrams import *
from chemgrams.logger import get_logger
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.sascorer import sascorer

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.klm')")
logger.info("num_chars=100, text_seed='<s>'")

vocab = get_arpa_vocab('../resources/postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.klm', vocab)


def log_top_best_with_props(pairs, props, top_n, lgg):
    all_best = reversed(list(reversed(sorted(pairs.items(), key=lambda kv: kv[1][0])))[:top_n])
    for i, ab in enumerate(all_best):
        molprops = props[ab[0]]
        lgg.info("%d. %s, %s, QED: %s, SA: %s, J: %s, mean similarity.: %s, max similarity.: %s, most similar: %s" %
                 (top_n - i, ab[0], str(ab[1]), molprops[0], molprops[1], molprops[2], molprops[3], molprops[4], molprops[5]))


logger.info("reading submissions...")
with open("../models/postera-covid-submissions-2020-03-28.txt") as f:
    submissions = f.readlines()
submissions = [Chem.MolFromSmiles(x.strip()) for x in submissions]
logger.info("finished reading %s submissions" % len(submissions))

logger.info("reading active site fragments...")
with open("../models/postera-covid-active-site-frags-2020-03-25.txt") as f:
    active_site_frags = f.readlines()
active_site_frags = [Chem.MolFromSmiles(x.strip()) for x in active_site_frags]
logger.info("finished reading %s active site fragments" % len(active_site_frags))

submissions.extend(active_site_frags)
logger.info("total number of submissions and active site fragments: %s" % len(submissions))

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

mol_props = {}
all_smiles = {}
num_valid = 0

for i in range(100000):
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        mol = Chem.MolFromSmiles(sanitized)

        # validate the geometry; this will raise an exception if the geometry is non-sensical
        mol2 = Chem.AddHs(mol, explicitOnly=False)
        AllChem.EmbedMolecule(mol2)
        AllChem.UFFOptimizeMolecule(mol2)

        num_valid += 1

        qedscore = QED.qed(mol)
        similarities = []
        most_similar_submission_mol = None
        max_similarity = 0.0
        for submission in submissions:
            similarity = TanimotoScorer.score_mols(submission, mol)
            similarities.append(similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_submission_mol = submission
        mean_distance = np.mean(similarities)

        sa_score = sascorer.calculateScore(mol)
        sa_score = (10.0 - sa_score) / 10.0

        tot_score = 0.5*qedscore + 0.5*sa_score

        if max_similarity >= 0.9:
            logger.info("discarding %s (too similar to submission or fragment: %s)" % (sanitized, Chem.MolToSmiles(most_similar_submission_mol)))
            continue

        all_smiles[sanitized] = (tot_score, generated)

        mol_props[sanitized] = (qedscore, sa_score, tot_score, mean_distance, max_similarity, Chem.MolToSmiles(most_similar_submission_mol))

        if current_best_score is None or beats_current(qedscore):
            current_best_score = qedscore
            current_best_smiles = sanitized

    except Exception as e:
        pass

    if (i+1) % 5000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        log_top_best_with_props(all_smiles, mol_props, 15, logger)

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))

log_top_best_with_props(all_smiles, mol_props, 1000, logger)
