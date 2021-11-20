import os
import time
from pathlib import Path
import shutil
from threading import Timer
import numpy as np

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, StopTreeSearch, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating, DeepSMILESTokenizer
from chemgrams.logger import get_logger, log_top_best
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.sascorer import sascorer
from chemgrams.cyclescorer import CycleScorer
from chemgrams.training import KenLMTrainer

from openbabel import pybel
from deepsmiles import Converter
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
logger = get_logger('chemgrams.log')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel('../models/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)")
logger.info("width = 12, max_depth = 50, start_state = ['<s>'], c = 2")
logger.info("score: -1.0 if invalid; -1.0 if seen in all smiles; tanimoto distance from abilify if valid")
logger.info("LanguageModelMCTSWithPUCTTerminating")
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
num_simulations = 15000000  # much more than 8 hours

all_smiles = {}

for n in range(num_iterations):
    num_valid = 0
    simulations = 0

    width = 12
    max_depth = 50
    start_state = ["<s>"]
    c = 2
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

    def eval_function(text):
        global simulations, num_valid, all_smiles, elapsed

        if elapsed >= TIME_PER_ITERATION:
            raise StopTreeSearch()

        simulations += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: raise Exception
        except Exception:
            elapsed = time.time() - start
            return -1.0

        num_valid += 1

        if smiles in seen:
            score = -1.0
        else:
            # synthetic accessibility score is a number between 1 (easy to make) and 10 (very difficult to make)
            sascore = sascorer.calculateScore(mol) / 10.

            # cycle score, squashed between 0 and 1
            cyclescore = cycle_scorer.score_mol(mol)
            cyclescore = cyclescore / (1 + cyclescore)

            distance_score = distance_scorer.score_mol(mol)

            score = (0.75 * distance_score) + (0.15 * (1 - sascore)) + (0.10 * (1 - cyclescore))

            seen.add(smiles)
            all_smiles[smiles] = (score, generated)
            if distance_score == 1.0:
                logger.info("FOUND!")

        # rescale score from [0,1] to [-1,1]
        ret_score = (score * 2) + (-1) if score >= 0. else score

        elapsed = time.time() - start
        return ret_score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    try:
        mcts.search(state, num_simulations)
    except StopTreeSearch:
        pass

    t.cancel()
    end = time.time()

    logger.info("--done--")
    logger.info("num simulations: %s" % simulations)
    logger.info("num valid (in this iteration): %d" % num_valid)
    logger.info("num unique (over all iterations): %s" % len(all_smiles))
    logger.info("num unique (in this iteration): %s" % len(seen))

    best = mcts.get_best_sequence()
    generated_text = ''.join(best[0])
    logger.info("best generated text: %s" % generated_text)
    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (smiles, distance_scorer.score(smiles), str((end - start))))

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
