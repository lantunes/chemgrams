import os

import pybel
from deepsmiles import Converter
from rdkit import rdBase

from chemgrams import *
from chemgrams.logger import get_logger

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
from molexit import KenLMTrainer
logger = get_logger('chemgrams.log')

MOLEXIT_DIR = "../models/molexit-postera"


logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.klm')")
logger.info("width = 24, max_depth = 100, start_state = ['<s>'], c = 5")
logger.info("ChemProp Docking Scorer (65%) + QED (35%)")
logger.info("LanguageModelMCTSWithPUCTTerminating")
logger.info("num_iterations = 15")
logger.info("simulations_per_iteration = 50000")
logger.info("keep_top_n = 5000")

logger.info("loading language model...")

vocab = get_arpa_vocab('../resources/postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.klm', vocab)

converter = Converter(rings=True, branches=True)
env = os.environ.copy()
env["PATH"] = "/Users/luis/kenlm/build/bin:" + env["PATH"]
lm_trainer = KenLMTrainer('../utils/train_kenlm.sh', env)


def smiles_to_deepsmiles(smiles):
    canonical = pybel.readstring("smi", smiles).write("can").strip()
    return converter.encode(canonical)

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

num_iterations = 15
simulations_per_iteration = 500000
keep_top_n = 2500

mol_props = {}
all_smiles = {}
sanitized_to_final = {}

for n in range(num_iterations):

    num_simulations = simulations_per_iteration
    width = 24
    max_depth = 100
    start_state = ["<s>"]
    c = 5

    num_valid = 0
    i = 0

    def eval_function(text):
        global i, num_valid, all_smiles, mol_props
        i += 1

        generated = ''.join(text)

        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

            mol = Chem.MolFromSmiles(sanitized)

            Utils.validate_geometry(mol)

            num_valid += 1

            if sanitized in all_smiles:
                raise Exception("already generated")

            canon_smi = Chem.CanonSmiles(sanitized)
            if canon_smi in existing_set:
                logger.info("discarding %s (already exists in submissions or fragments)" % sanitized)
                raise Exception("duplicate")

            # filter out mols that are fragment-like by checking mol. wt.
            mw = Utils.get_mw(mol)
            if mw < 349.0:
                raise Exception("too small")

            # assign some stereochemistry if there are stereocenters
            final_smi = Utils.get_first_stereoisomer(mol)

            sanitized_to_final[sanitized] = final_smi

            docking_score, norm_docking_score = Utils.get_chemprop_docking_score_normalized(docking_scorer, final_smi,
                                                                                            docking_min, docking_max)

            qedscore = Utils.get_qed(mol)

            tot_score = 0.65 * norm_docking_score + 0.35 * qedscore

            all_smiles[final_smi] = (tot_score, generated)

            mol_props[final_smi] = {"docking": docking_score, "qed": qedscore}

            score = tot_score

        except Exception:
            score = -1.0

        if (i + 1) % 1000 == 0:
            logger.info("--iteration: %d--" % (i + 1))
            logger.info("num valid: %d" % num_valid)
            Utils.log_top_best(all_smiles, mol_props, 5, logger)

        return score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")
    mcts.search(state, num_simulations)

    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)

    best = mcts.get_best_sequence()
    generated_text = ''.join(best[0])
    logger.info("best generated text: %s" % generated_text)
    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    decoded_smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
    smiles = sanitized_to_final[decoded_smiles]
    logger.info("best SMILES: %s, J: %s" % (smiles, str(all_smiles[smiles])))

    Utils.log_top_best_with_descriptors(all_smiles, mol_props, 5, submissions, logger)

    logger.info("writing dataset...")
    name = 'molexit-%d' % n
    dataset = '%s/%s.txt' % (MOLEXIT_DIR, name)
    with open(dataset, 'w') as f:
        for smi in list(reversed(sorted(all_smiles.items(), key=lambda kv: kv[1][0])))[:keep_top_n]:
            dsmi = smiles_to_deepsmiles(smi[0].strip())
            tok = DeepSMILESTokenizer(dsmi)
            tokens = tok.get_tokens()
            f.write(' '.join([t.value for t in tokens]))
            f.write("\n")

    logger.info('training new LM...')
    lm_trainer.train(dataset, MOLEXIT_DIR, name)

    vocab = get_arpa_vocab('%s/%s.arpa' % (MOLEXIT_DIR, name))
    lm = KenLMDeepSMILESLanguageModel('%s/%s.klm' % (MOLEXIT_DIR, name), vocab)
