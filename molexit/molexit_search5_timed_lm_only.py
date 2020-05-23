import os
import time
from threading import Timer

import pybel
from deepsmiles import Converter
from rdkit import rdBase

from chemgrams import *
from chemgrams.logger import get_logger, log_top_best
from chemgrams.tanimotoscorer import TanimotoScorer

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
from chemgrams.sascorer import sascorer
from chemgrams.cyclescorer import CycleScorer
from chemgrams.training import KenLMTrainer
logger = get_logger('chemgrams.log')
from pathlib import Path
import shutil
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel('../models/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)")
logger.info("TanimotoScorer(abilify, radius=6)")
logger.info("num_iterations = 100")
logger.info("time per iteration = 45 min.")
logger.info("keep_top_n = 20000")

vocab = get_arpa_vocab('../models/chembl_25_deepsmiles_klm_10gram_200503.arpa')
lm = KenLMDeepSMILESLanguageModel('../models/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
distance_scorer = TanimotoScorer(abilify, radius=6)

cycle_scorer = CycleScorer()

converter = Converter(rings=True, branches=True)
env = os.environ.copy()
env["PATH"] = "/Users/luis/kenlm/build/bin:" + env["PATH"]
lm_trainer = KenLMTrainer(env)


def smiles_to_deepsmiles(smiles):
    canonical = pybel.readstring("smi", smiles).write("can").strip()
    return converter.encode(canonical)

logger.info("deleting any existing molexit directory, and creating a new one...")
path = Path("../models/molexit/")
if os.path.exists(path) and os.path.isdir(path):
    shutil.rmtree(path)
path.mkdir(parents=True, exist_ok=True)

num_iterations = 100
keep_top_n = 20000
TIME_PER_ITERATION = 45*60  # 45 minutes in seconds
LOG_INTERVAL = 5*60.0  # 5 minutes in seconds

all_smiles = {}

for n in range(num_iterations):
    num_valid = 0
    simulations = 0
    current_best_score = None
    current_best_smiles = None
    beats_current = lambda sc: sc > current_best_score
    seen = set()

    logger.info("searching...")

    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num simulations: %s" % simulations)
        logger.info("num valid (in this iteration): %d" % num_valid)
        logger.info("num unique (over all iterations): %s" % len(all_smiles))
        logger.info("num unique (in this iteration): %s" % len(seen))
        log_top_best(all_smiles, 5, logger)
        t = Timer(LOG_INTERVAL, log_progress)
        t.start()
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()

    start = time.time()
    elapsed = time.time() - start
    while elapsed < TIME_PER_ITERATION:
        simulations += 1
        try:
            generated = lm.generate(num_chars=100, text_seed='<s>')

            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)
            mol = Chem.MolFromSmiles(sanitized)

            num_valid += 1

            # synthetic accessibility score is a number between 1 (easy to make) and 10 (very difficult to make)
            sascore = sascorer.calculateScore(mol) / 10.
            # cycle score, squashed between 0 and 1
            cyclescore = cycle_scorer.score_mol(mol)
            cyclescore = cyclescore / (1 + cyclescore)
            distance_score = distance_scorer.score_mol(mol)
            score = (0.75 * distance_score) + (0.15 * (1 - sascore)) + (0.10 * (1 - cyclescore))

            all_smiles[sanitized] = (score, generated)
            seen.add(sanitized)
            if distance_score == 1.0:
                logger.info("FOUND!")

            if current_best_score is None or beats_current(score):
                current_best_score = score
                current_best_smiles = sanitized

        except Exception as e:
            pass

        elapsed = time.time() - start

    t.cancel()
    end = time.time()
    logger.info("--done--")
    logger.info("num simulations: %s" % simulations)
    logger.info("num valid (in this iteration): %d" % num_valid)
    logger.info("num unique (over all iterations): %s" % len(all_smiles))
    logger.info("num unique (in this iteration): %s" % len(seen))
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (current_best_smiles, current_best_score, str((end - start))))

    log_top_best(all_smiles, 5, logger)

    logger.info("writing dataset...")
    name = 'molexit-%d' % n
    dataset = '../models/molexit/%s.txt' % name
    dataset_scores = []
    with open(dataset, 'w') as f:
        for smi in list(reversed(sorted(all_smiles.items(), key=lambda kv: kv[1][0])))[:keep_top_n]:
            dsmi = smiles_to_deepsmiles(smi[0].strip())
            tok = DeepSMILESTokenizer(dsmi)
            tokens = tok.get_tokens()
            f.write(' '.join([t.value for t in tokens]))
            f.write("\n")
            dataset_scores.append(smi[1][0])

    logger.info('dataset: size: %s, mean score: %s' % (len(dataset_scores), np.mean(dataset_scores)))
    logger.info('training new LM...')
    lm_trainer.train(10, dataset, '../models/molexit', name)

    vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
    lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)
