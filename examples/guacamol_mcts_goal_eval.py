import os

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating

from guacamol.assess_goal_directed_generation import GoalDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.helpers import setup_default_logger


class ChemgramsMCTSGoalDirectedGenerator(GoalDirectedGenerator):

    def __init__(self):
        self.vocab = get_arpa_vocab('../resources/chembl_25_deepsmiles_klm_10gram_200503.arpa')
        self.lm = KenLMDeepSMILESLanguageModel('../resources/chembl_25_deepsmiles_klm_10gram_200503.klm', self.vocab)
        self.best_smiles = None
        self.best_score = -1.0

    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):
        print("generating %s samples..." % number_molecules)
        all_smiles = set()
        width = 24
        max_depth = 100
        c = 5
        num_simulations = 10000

        self.best_smiles = None
        self.best_score = -1.0

        def eval_function(text):

            generated = ''.join(text)
            try:
                decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
                smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
            except Exception:
                return -1.0

            if smiles in all_smiles:
                score = -1.0
            else:
                score = scoring_function.score(smiles)
                all_smiles.add(smiles)

            if self.best_score < score:
                self.best_score = score
                self.best_smiles = smiles

            return score

        mcts = LanguageModelMCTSWithPUCTTerminating(self.lm, width, max_depth, eval_function, cpuct=c,
                                                    terminating_symbol='</s>')
        mcts.search(["<s>"], num_simulations)

        return [self.best_smiles]


if __name__ == '__main__':
    setup_default_logger()

    generator = ChemgramsMCTSGoalDirectedGenerator()

    json_file_path = os.path.join('../models', 'goal_directed_learning_results.json')

    assess_goal_directed_generation(generator,
                                    json_output_file=json_file_path,
                                    benchmark_version='v2')
