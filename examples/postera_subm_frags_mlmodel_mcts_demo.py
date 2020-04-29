import os

from rdkit import rdBase

from chemgrams import *
from chemgrams.logger import get_logger

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')


if __name__ == '__main__':

    logger.info(os.path.basename(__file__))
    logger.info("KenLMDeepSMILESLanguageModel(n=6, 'postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.klm')")
    logger.info("width = 24, max_depth = 100, start_state = ['<s>'], c = 5")
    logger.info("LanguageModelMCTSWithPUCTTerminating")
    logger.info("ChemProp Docking Scorer (65%) + QED (35%)")

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../resources/postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/postera_covid_subm_frags_2020_03_27_deepsmiles_klm_6gram.klm', vocab)

    num_simulations = 3000000  # ~8 hours
    width = 24
    max_depth = 100
    start_state = ["<s>"]
    c = 5

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

    mol_props = {}
    all_smiles = {}
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
            sanitized = Utils.get_first_stereoisomer(mol)

            docking_score, norm_docking_score = Utils.get_chemprop_docking_score_normalized(docking_scorer, sanitized,
                                                                                            docking_min, docking_max)

            qedscore = Utils.get_qed(mol)

            tot_score = 0.65 * norm_docking_score + 0.35 * qedscore

            all_smiles[sanitized] = (tot_score, generated)

            mol_props[sanitized] = {"docking": docking_score, "qed": qedscore}

            score = tot_score

        except Exception:
            score = -1.0

        if (i + 1) % 1000 == 0:
            logger.info("--iteration: %d--" % (i + 1))
            logger.info("num valid: %d" % num_valid)
            Utils.log_top_best(all_smiles, mol_props, 15, logger)

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
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    logger.info("best SMILES: %s, J: %s" % (smiles, str(all_smiles[smiles])))

    Utils.log_top_best_with_descriptors(all_smiles, mol_props, 1000, submissions, logger)
