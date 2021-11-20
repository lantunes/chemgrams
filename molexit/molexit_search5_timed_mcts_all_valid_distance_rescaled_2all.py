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
from chemgrams.training import KenLMTrainer

from openbabel import pybel
from deepsmiles import Converter
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
logger = get_logger('chemgrams.log')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)")
logger.info("width = 12, max_depth = 50, start_state = ['<s>'], c = 2")
logger.info("score: -1.0 if invalid; -1.0 if seen in iteration; tanimoto distance from abilify if valid; rescaling from [0,1] to [-1,1]")
logger.info("LanguageModelMCTSWithPUCTTerminating")
logger.info("TanimotoScorer(abilify, radius=6); distance only (no SA or cycle scoring)")
logger.info("num_iterations = 100")
logger.info("time per iteration = 45 min.")
logger.info("keep all valid in current iteration (including duplicates)")

vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
distance_scorer = TanimotoScorer(abilify, radius=6)

# celecoxib = "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"
# distance_scorer = TanimotoScorer(celecoxib, radius=6)

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
TIME_PER_ITERATION = 240*60 #45*60  # 45 minutes in seconds
LOG_INTERVAL = 5*60.0  # 5 minutes in seconds
num_simulations = 15000000  # much more than 8 hours

all_unique = {}

c_sched = [5] #[2, 5, 8, 11, 15]
c_default = 5
max_gen = 300000

for n in range(num_iterations):
    all_valid = []
    simulations = 0

    width = 12
    max_depth = 50
    start_state = ["<s>"]
    c = c_sched[n] if n < len(c_sched) else c_default
    seen = set()

    current_best_score = None
    current_best_smiles = None
    beats_current = lambda sc: sc > current_best_score

    time_limit = TIME_PER_ITERATION #if n == 0 else 7*60

    logger.info("searching...")

    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num simulations: %s" % simulations)
        logger.info("num valid (in this iteration): %d" % len(all_valid))
        logger.info("num unique (over all iterations): %s" % len(all_unique))
        logger.info("num unique (in this iteration): %s" % len(seen))
        log_top_best(all_unique, 5, logger)
        t = Timer(LOG_INTERVAL, log_progress)
        t.start()
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()

    start = time.time()
    elapsed = time.time() - start

    def eval_function(text):
        global simulations, all_unique, elapsed, current_best_score, current_best_smiles, beats_current

        if elapsed >= time_limit or len(all_valid) == max_gen:
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

        score = distance_scorer.score_mol(mol)

        if current_best_score is None or beats_current(score):
            current_best_score = score
            current_best_smiles = smiles

        if score == 1.0:
            logger.info("FOUND!")

        ret_score = -1.0 if smiles in seen else score

        # rescale score from [0,1] to [-1,1]
        ret_score = (ret_score * 2) + (-1) if ret_score >= 0. else ret_score

        all_unique[smiles] = (score, generated)
        all_valid.append((smiles, score))
        seen.add(smiles)

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
    logger.info("num valid (in this iteration): %d" % len(all_valid))
    logger.info("num unique (over all iterations): %s" % len(all_unique))
    logger.info("num unique (in this iteration): %s" % len(seen))
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (current_best_smiles, current_best_score, str((end - start))))

    log_top_best(all_unique, 5, logger)

    logger.info("writing dataset...")
    name = 'molexit-%d' % n
    dataset = '../models/molexit/%s.txt' % name
    dataset_scores = []
    with open(dataset, 'w') as f:
        for smi in all_valid:
            try:
                dsmi = smiles_to_deepsmiles(smi[0].strip())
                tok = DeepSMILESTokenizer(dsmi)
                tokens = tok.get_tokens()
                f.write(' '.join([t.value for t in tokens]))
                f.write("\n")
                dataset_scores.append(smi[1])
            except Exception:
                pass

    with open('../models/molexit/%s-iteration-valid-all-scores.txt' % name, 'w') as f:
        [f.write('%s\n' % s) for s in dataset_scores]

    logger.info('dataset: size: %s, mean score: %s, std: %s' % (len(dataset_scores), np.mean(dataset_scores), np.std(dataset_scores)))

    logger.info('training new LM...')
    lm_trainer.train(10, dataset, '../models/molexit', name)

    vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
    lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)
