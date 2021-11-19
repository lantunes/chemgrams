import os
import time
from pathlib import Path
import shutil

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating, DeepSMILESTokenizer
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.logger import get_logger, log_top_best
from chemgrams.sascorer import sascorer
from chemgrams.cyclescorer import CycleScorer
from chemgrams.training import KenLMTrainer

import pybel
from deepsmiles import Converter
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
logger = get_logger('chemgrams.log')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info(os.path.basename(__file__))
logger.info("KenLMDeepSMILESLanguageModel('../resources/zinc12_fragments_deepsmiles_klm_6gram_190421.klm', vocab)")
logger.info("width = 12, max_depth = 50, start_state = ['<s>'], c = 5")
logger.info("score: -1.0 if invalid; -1.0 if seen previously; tanimoto distance from abilify if valid")
logger.info("LanguageModelMCTSWithPUCTTerminating")
logger.info("TanimotoScorer(abilify, radius=6)")
logger.info("num_iterations = 300")
logger.info("simulations_per_iteration = 200000")
logger.info("keep_top_n = 50000")

logger.info("loading language model...")

vocab = get_arpa_vocab('../resources/zinc12_fragments_deepsmiles_klm_6gram_190421.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/zinc12_fragments_deepsmiles_klm_6gram_190421.klm', vocab)

abilify = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
distance_scorer = TanimotoScorer(abilify, radius=6)

cycle_scorer = CycleScorer()

converter = Converter(rings=True, branches=True)
env = os.environ.copy()
env["PATH"] = "/Users/luis/kenlm/build/bin:" + env["PATH"]
lm_trainer = KenLMTrainer(env)


def log_best(j, all_best, n_valid, lggr):
    if j % 10000 == 0:
        lggr.info("--iteration: %d--" % j)
        lggr.info("num valid: %d" % n_valid)
        log_top_best(all_best, 5, lggr)


def smiles_to_deepsmiles(smiles):
    canonical = pybel.readstring("smi", smiles).write("can").strip()  # TODO do we need to canonicalize?
    return converter.encode(canonical)

logger.info("deleting any existing molexit directory, and creating a new one...")
path = Path("../models/molexit/")
if os.path.exists(path) and os.path.isdir(path):
    shutil.rmtree(path)
path.mkdir(parents=True, exist_ok=True)

num_iterations = 300
simulations_per_iteration = 200000
keep_top_n = 50000

all_smiles = {}

for n in range(num_iterations):

    num_simulations = simulations_per_iteration
    width = 12
    max_depth = 50
    start_state = ["<s>"]
    c = 5

    seen_smiles = set()

    num_valid = 0
    i = 0

    def eval_function(text):
        global i, num_valid, all_smiles
        i += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            log_best(i, all_smiles, num_valid, logger)
            return -1.0

        num_valid += 1

        if smiles in seen_smiles:
            score = -1.0
        else:
            # synthetic accessibility score is a number between 1 (easy to make) and 10 (very difficult to make)
            sascore = sascorer.calculateScore(Chem.MolFromSmiles(smiles)) / 10.

            # cycle score, squashed between 0 and 1
            cyclescore = cycle_scorer.score(smiles)
            cyclescore = cyclescore / (1 + cyclescore)

            distance_score = distance_scorer.score(smiles)

            score = (0.75*distance_score) + (0.15*(1-sascore)) + (0.10*(1-cyclescore))

            seen_smiles.add(smiles)
            all_smiles[smiles] = (score, generated)

        logger.debug("%s, %s" % (smiles, str(score)))
        log_best(i, all_smiles, num_valid, logger)
        return score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")
    start = time.time()
    mcts.search(state, num_simulations)
    end = time.time()

    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)

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
    with open(dataset, 'w') as f:
        for smi in list(reversed(sorted(all_smiles.items(), key=lambda kv: kv[1][0])))[:keep_top_n]:
            dsmi = smiles_to_deepsmiles(smi[0].strip())
            tok = DeepSMILESTokenizer(dsmi)
            tokens = tok.get_tokens()
            f.write(' '.join([t.value for t in tokens]))
            f.write("\n")

    logger.info('training new LM...')
    lm_trainer.train(6, dataset, '../models/molexit', name)

    vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
    lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)
