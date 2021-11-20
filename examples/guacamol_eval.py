import os

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils

from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger


class ChemgramsSmilesSampler(DistributionMatchingGenerator):

    def __init__(self):
        self.vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
        self.lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', self.vocab)

    def generate(self, number_samples):
        print("generating %s samples..." % number_samples)
        samples = []

        for n in range(number_samples):
            try:
                generated = self.lm.generate(num_chars=100, text_seed='<s>')
                decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
                sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)
            except Exception:
                sanitized = "invalid"
            samples.append(sanitized)

        return samples


if __name__ == '__main__':
    setup_default_logger()

    generator = ChemgramsSmilesSampler()

    chembl_training_file = os.path.join('../models/guacamol_data/guacamol_v1_all.smiles')
    json_file_path = os.path.join('../models', 'distribution_learning_results.json')

    assess_distribution_learning(generator,
                                 chembl_training_file=chembl_training_file,
                                 json_output_file=json_file_path,
                                 benchmark_version='v2')
