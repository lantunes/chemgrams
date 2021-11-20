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

from openbabel import pybel
from deepsmiles import Converter
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
from chemgrams.training import KenLMTrainer
logger = get_logger('chemgrams.log')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)")
logger.info("width = 12, max_depth = 50, start_state = ['<s>'], c = 5")
logger.info("score: -1.0 if invalid; -1.0 if seen in iteration; tanimoto distance from abilify if valid; rescaling from [0,1] to [-1,1]")
logger.info("score = log(prob_prior(m)) + sigma*distance_score; sigma = 10")
logger.info("LanguageModelMCTSWithPUCTTerminating")
logger.info("TanimotoScorer(abilify, radius=6); distance only (no SA or cycle scoring)")
logger.info("num_iterations = 100")
logger.info("time per iteration = 45 min.")
logger.info("keep_top_n = 250000 of all (including duplicates)")

vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
prior = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

lm = prior

abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
distance_scorer = TanimotoScorer(abilify, radius=6)

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
keep_top_n = 250000
TIME_PER_ITERATION = 45*60  # 45 minutes in seconds
LOG_INTERVAL = 5*60.0  # 5 minutes in seconds
num_simulations = 15000000  # much more than 8 hours

all_unique = {}
all_valid = []

for n in range(num_iterations):
    num_valid = 0
    simulations = 0

    width = 12
    max_depth = 50
    start_state = ["<s>"]
    c = 5
    sigma = 10
    seen = set()

    current_best_score = None
    current_best_smiles = None
    beats_current = lambda sc: sc > current_best_score

    logger.info("searching...")

    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num simulations: %s" % simulations)
        logger.info("num valid (in this iteration): %d" % num_valid)
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
        global simulations, num_valid, all_unique, elapsed, current_best_score, current_best_smiles, beats_current

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

        distance_score = distance_scorer.score_mol(mol)
        if distance_score == 1.0:
            logger.info("FOUND!")

        # As in "Molecular de-novo design through deep reinforcement learning", by Olivecrona et al., we are adding
        #  the prior's log probability of the generated sequence to the score.
        prior_log_prob = prior.log_prob(
            DeepSMILESLanguageModelUtils.extract_sentence(text, join_on=' ', start='<s>', end='</s>'))

        # tot_score = prior_log_prob + sigma * ((distance_score * 2) + (-1))  # rescale the distance score from [0,1] to [-1,1]
        tot_score = prior_log_prob + sigma * distance_score

        # rescale the score
        # in practice, the log probs are rarely less than -45; so the min tot_score can be: -45 + (sigma*-1.0)
        rescale_min = -45 - sigma
        if tot_score < rescale_min:
            logger.info("WARNING: total score lower than %s" % rescale_min)
        # because probabilities are in the range [0,1], the max log prob is log(1) i.e. 0
        #  so the max tot_score can be: 0 + sigma*1.0
        rescale_max = sigma
        # scaling x into [a,b]: (b-a)*((x - min(x))/(max(x) - min(x))+a
        ret_score = (1 - (-1)) * ((tot_score - rescale_min) / (rescale_max - rescale_min)) + (-1)

        ret_score = -1.0 if smiles in seen else ret_score

        if current_best_score is None or beats_current(distance_score):
            current_best_score = distance_score
            current_best_smiles = smiles

        all_unique[smiles] = (distance_score, generated)
        all_valid.append((smiles, distance_score))
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
    logger.info("num valid (in this iteration): %d" % num_valid)
    logger.info("num valid (over all iterations): %d" % len(all_valid))
    logger.info("num unique (over all iterations): %s" % len(all_unique))
    logger.info("num unique (in this iteration): %s" % len(seen))
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (current_best_smiles, current_best_score, str((end - start))))

    log_top_best(all_unique, 5, logger)

    logger.info("writing dataset...")
    name = 'molexit-%d' % n
    dataset = '../models/molexit/%s.txt' % name
    dataset_scores = []
    with open(dataset, 'w') as f:
        for smi in list(reversed(sorted(all_valid, key=lambda i: i[1])))[:keep_top_n]:
            try:
                dsmi = smiles_to_deepsmiles(smi[0].strip())
                tok = DeepSMILESTokenizer(dsmi)
                tokens = tok.get_tokens()
                f.write(' '.join([t.value for t in tokens]))
                f.write("\n")
                dataset_scores.append(smi[1])
            except Exception:
                pass

    logger.info('dataset: size: %s, mean score: %s' % (len(dataset_scores), np.mean(dataset_scores)))
    logger.info('training new LM...')
    lm_trainer.train(10, dataset, '../models/molexit', name)

    vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
    lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)
