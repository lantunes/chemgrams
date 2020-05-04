import subprocess
import os.path as op


class KenLMTrainer:
    def __init__(self, env=None):
        self._shell_script_path = op.join(op.dirname(__file__), "train_kenlm.sh")
        self._env = env

    def train(self, order, corpus_file_path, output_dir, output_file_name):
        ret = subprocess.call(['sh', self._shell_script_path, str(order), corpus_file_path, output_dir, output_file_name], env=self._env)
        if ret != 0:
            raise Exception("error code received training LM: %d" % ret)
