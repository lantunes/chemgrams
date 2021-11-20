import os

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating

from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger


class ChemgramsMCTSSmilesSampler(DistributionMatchingGenerator):

    def __init__(self):
        self.vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
        self.lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', self.vocab)

    def generate(self, number_samples):
        print("generating %s samples..." % number_samples)
        all_smiles = set()
        samples = []
        width = 24
        max_depth = 100
        c = 5

        def eval_function(text):

            generated = ''.join(text)
            try:
                decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
                smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
            except Exception:
                samples.append("invalid")
                return -1.0

            samples.append(smiles)

            if smiles in all_smiles:
                score = -1.0
            else:
                score = 1.0
                all_smiles.add(smiles)

            return score

        mcts = LanguageModelMCTSWithPUCTTerminating(self.lm, width, max_depth, eval_function, cpuct=c,
                                                    terminating_symbol='</s>')
        mcts.search(["<s>"], number_samples)

        return samples


if __name__ == '__main__':
    setup_default_logger()

    generator = ChemgramsMCTSSmilesSampler()

    chembl_training_file = os.path.join('../models/guacamol_data/guacamol_v1_all.smiles')
    json_file_path = os.path.join('../models', 'distribution_learning_results.json')

    assess_distribution_learning(generator,
                                 chembl_training_file=chembl_training_file,
                                 json_output_file=json_file_path,
                                 benchmark_version='v2')
