import os
import time
from pathlib import Path
import shutil

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, DeepSMILESTokenizer
from chemgrams.tanimotoscorer import TanimotoScorer
from chemgrams.logger import get_logger, log_top_best
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
logger.info("KenLMDeepSMILESLanguageModel('../resources/zinc12_fragments_deepsmiles_klm_10gram_200502.klm', vocab)")
logger.info("TanimotoScorer(abilify, radius=6)")
logger.info("num_iterations = 100")
logger.info("attempts_per_iteration = 400000")
logger.info("keep_top_n = 20000")

logger.info("loading language model...")

vocab = get_arpa_vocab('../resources/zinc12_fragments_deepsmiles_klm_10gram_200502.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/zinc12_fragments_deepsmiles_klm_10gram_200502.klm', vocab)

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
attempts_per_iteration = 400000
keep_top_n = 20000

all_smiles = {}

for n in range(num_iterations):

    num_attempts = attempts_per_iteration
    num_valid = 0
    current_best_score = None
    current_best_smiles = None
    beats_current = lambda score: score > current_best_score
    start = time.time()

    logger.info("beginning search...")
    for i in range(num_attempts):
        try:
            generated = lm.generate(num_chars=100, text_seed='<s>')

            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

            num_valid += 1

            # synthetic accessibility score is a number between 1 (easy to make) and 10 (very difficult to make)
            sascore = sascorer.calculateScore(Chem.MolFromSmiles(sanitized)) / 10.
            # cycle score, squashed between 0 and 1
            cyclescore = cycle_scorer.score(sanitized)
            cyclescore = cyclescore / (1 + cyclescore)
            distance_score = distance_scorer.score(sanitized)
            score = (0.75 * distance_score) + (0.15 * (1 - sascore)) + (0.10 * (1 - cyclescore))

            all_smiles[sanitized] = (score, generated)

            if current_best_score is None or beats_current(score):
                current_best_score = score
                current_best_smiles = sanitized

        except Exception as e:
            pass

        if (i + 1) % 50000 == 0:
            logger.info("--iteration: %d--" % (i + 1))
            logger.info("num valid (in this iteration): %d" % num_valid)
            logger.info("num unique (over all iterations): %s" % len(all_smiles))
            log_top_best(all_smiles, 5, logger)

    end = time.time()
    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)
    logger.info("best SMILES: %s, J: %s (%s seconds)" % (current_best_smiles, current_best_score, str((end - start))))

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
    lm_trainer.train(10, dataset, '../models/molexit', name)

    vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
    lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)
