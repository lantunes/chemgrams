import subprocess


class KenLMTrainer:
    def __init__(self, shell_script_path, env=None):
        """
        :param shell_script_path: the path to the shell script (e.g. "../utils/train_kenlm.sh")
        """
        self._shell_script_path = shell_script_path
        self._env = env

    def train(self, corpus_file_path, output_dir, output_file_name):
        ret = subprocess.call(['sh', self._shell_script_path, corpus_file_path, output_dir, output_file_name], env=self._env)
        if ret != 0:
            raise Exception("error code received training LM: %d" % ret)
