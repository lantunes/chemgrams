import csv
import subprocess
import tempfile

CHEMPROP_DIR = "/Users/luis/chemprop"
PYTHON_ENV_PATH = "/Users/luis/anaconda/envs/chemprop/bin/"
# CHECKPOINT_DIR = "/Users/luis/chemprop/postera_hybrid2_checkpoints6"
CHECKPOINT_DIR = "/Users/luis/chemprop/postera_hybrid2_checkpoints8"


class ChemPropDockingScorer:
    def __init__(self,
                 chemprop_dir=CHEMPROP_DIR,
                 python_env_path=PYTHON_ENV_PATH,
                 checkpoint_dir=CHECKPOINT_DIR,
                 print_stdout=False):
        self.chemprop_dir = chemprop_dir
        self.python_env_path = python_env_path
        self.checkpoint_dir = checkpoint_dir
        self.print_stdout = print_stdout

    def score(self, smiles_string):
        # generate the input file
        test_file = tempfile.NamedTemporaryFile()
        with open(test_file.name, "w") as f:
            f.writelines(["smiles\n", smiles_string])
            f.flush()
            f.close()

        pred_path = tempfile.NamedTemporaryFile()

        # call the model
        result = subprocess.run(["%s/python" % self.python_env_path, "predict.py", "--test_path", test_file.name,
                                 "--preds_path", pred_path.name, "--checkpoint_dir", self.checkpoint_dir,
                                 "--features_generator", "rdkit_2d_normalized", "--no_features_scaling"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.chemprop_dir)

        if self.print_stdout:
            print(result.stdout)

        # read the output file
        with open(pred_path.name, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None)
            score = float(next(reader, None)[1])

        return score


# scorer = ChemPropDockingScorer()
# print(scorer.score("CC(=O)NC1C=NC(NC2CCC(C3C=C(C)C=CC=3)CC2)=NC=1"))
