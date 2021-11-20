import os
import time
from pathlib import Path
import shutil
import numpy as np

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, DeepSMILESTokenizer
from chemgrams.training import KenLMTrainer

from guacamol.assess_goal_directed_generation import GoalDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.helpers import setup_default_logger

from openbabel import pybel

from deepsmiles import Converter


class ChemgramsGoalDirectedGenerator(GoalDirectedGenerator):

    def __init__(self, num_iterations, keep_top_n, time_per_iteration_minutes):
        self.num_iterations = num_iterations
        self.keep_top_n = keep_top_n
        self.time_per_iteration_minutes = time_per_iteration_minutes

        self.lm = None

        env = os.environ.copy()
        env["PATH"] = "/Users/luis/kenlm/build/bin:" + env["PATH"]
        self.lm_trainer = KenLMTrainer(env)

        self.converter = Converter(rings=True, branches=True)

    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):
        self.new_model_dir()

        vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
        self.lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', vocab)

        print("generating %s samples..." % number_molecules)
        smiles_and_scores = []

        TIME_PER_ITERATION = self.time_per_iteration_minutes * 60  # in seconds

        found = False
        for n in range(1, self.num_iterations+1):
            print("iteration %s" % n)
            num_valid = 0

            start = time.time()
            elapsed = time.time() - start
            while elapsed < TIME_PER_ITERATION:
                try:
                    generated = self.lm.generate(num_chars=100, text_seed='<s>')

                    decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
                    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

                    score = scoring_function.score(smiles)
                    num_valid += 1
                    smiles_and_scores.append((smiles, score))

                    if score == 1.0:
                        found = True
                        break

                except Exception:
                    pass
                elapsed = time.time() - start

            print("num valid: %s" % num_valid)

            if found:
                break

            self.retrain(n, self.keep_top_n, smiles_and_scores)

        return [pair[0] for pair in list(reversed(sorted(smiles_and_scores, key= lambda p: p[1])))[:number_molecules]]

    def new_model_dir(self):
        print("deleting any existing molexit directory, and creating a new one...")
        path = Path("../models/molexit/")
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def retrain(self, n, keep_top_n, smiles_and_scores):
        print("writing dataset...")
        name = 'molexit-%d' % n
        dataset = '../models/molexit/%s.txt' % name
        dataset_scores = []
        with open(dataset, 'w') as f:
            for smi, score in list(reversed(sorted(smiles_and_scores, key=lambda p: p[1])))[:keep_top_n]:
                dsmi = self.converter.encode(pybel.readstring("smi", smi.strip()).write("can").strip())
                tok = DeepSMILESTokenizer(dsmi)
                tokens = tok.get_tokens()
                f.write(' '.join([t.value for t in tokens]))
                f.write("\n")
                dataset_scores.append(score)

        print('dataset: size: %s, mean score: %s, max score: %s' %
              (len(dataset_scores), np.mean(dataset_scores), np.max(dataset_scores)))
        print('training new LM...')
        self.lm_trainer.train(10, dataset, '../models/molexit', name)

        vocab = get_arpa_vocab('../models/molexit/%s.arpa' % name)
        self.lm = KenLMDeepSMILESLanguageModel('../models/molexit/%s.klm' % name, vocab)


if __name__ == '__main__':
    setup_default_logger()

    generator = ChemgramsGoalDirectedGenerator(num_iterations=4,
                                               keep_top_n=1200,
                                               time_per_iteration_minutes=5)

    json_file_path = os.path.join('../models', 'goal_directed_learning_results.json')

    assess_goal_directed_generation(generator,
                                    json_output_file=json_file_path,
                                    benchmark_version='v2')
