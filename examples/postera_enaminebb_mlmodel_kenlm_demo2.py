import os

from rdkit import rdBase
from chemgrams import *
from chemgrams.logger import get_logger

from chemgrams.utils import Utils

from chemgrams.chemprop_docking_scorer import ChemPropDockingScorer

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.klm')")
logger.info("num_chars=100, text_seed='<s>'")

vocab = get_arpa_vocab('../resources/postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_zinc12_enaminebb_deepsmiles_klm_6gram.klm', vocab)

logger.info("reading submissions...")
with open("../models/postera-covid-submissions-2020-04-06.txt") as f:
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

existing_set = Utils.get_canon_set(submissions)

docking_scorer = ChemPropDockingScorer()
docking_max = 14.705297
docking_min = -11.795407

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

mol_props = {}
all_smiles = {}
num_valid = 0

logger.info("beginning generation...")

for i in range(100000):
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        mol = Chem.MolFromSmiles(sanitized)

        Utils.validate_geometry(mol)

        num_valid += 1

        canon_smi = Chem.CanonSmiles(sanitized)
        if canon_smi in existing_set:
            logger.info("discarding %s (already exists in submissions or fragments)")
            raise Exception("duplicate")

        # filter out mols that are fragment-like by checking mol. wt.
        mw = Utils.get_mw(mol)
        if mw < 349.0:
            raise Exception("too small")

        # assign some stereochemistry if there are stereocenters
        sanitized = Utils.get_first_stereoisomer(mol)

        docking_score, norm_docking_score = Utils.get_chemprop_docking_score_normalized(docking_scorer, sanitized, docking_min, docking_max)

        qedscore = Utils.get_qed(mol)

        tot_score = 0.65*norm_docking_score + 0.35*qedscore

        all_smiles[sanitized] = (tot_score, generated)

        mol_props[sanitized] = (docking_score, qedscore)

        if current_best_score is None or beats_current(tot_score):
            current_best_score = tot_score
            current_best_smiles = sanitized

    except Exception as e:
        pass

    if (i + 1) % 1000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        Utils.log_top_best(all_smiles, mol_props, 15, logger)

logger.info("--done--")
logger.info("num valid: %d" % num_valid)
logger.info("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))

Utils.log_top_best_with_descriptors(all_smiles, mol_props, 1000, submissions, logger)
