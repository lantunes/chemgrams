import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from chemgrams.sascorer import sascorer
from chemgrams.tanimotoscorer import TanimotoScorer


class Utils:

    @staticmethod
    def get_canon_set(all_mols):
        canon_set = set()
        for mol in all_mols:
            # TODO we might want to neutralize the molecule too
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
            if canon_smi not in canon_set:
                canon_set.add(canon_smi)
        return canon_set

    @staticmethod
    def get_mw(mol):
        return Descriptors.ExactMolWt(mol)

    @staticmethod
    def get_sa(mol):
        return sascorer.calculateScore(mol)

    @staticmethod
    def get_sa_normalized(mol):
        sa_score = sascorer.calculateScore(mol)
        sa_score = (10.0 - sa_score) / 10.0
        return sa_score

    @staticmethod
    def get_qed(mol):
        return QED.qed(mol)

    @staticmethod
    def get_chemprop_docking_score(scorer, smiles):
        return scorer.score(smiles)

    @staticmethod
    def get_chemprop_docking_score_normalized(scorer, smiles, min, max):
        docking_score = scorer.score(smiles)
        # rough normalization of docking score: x - min / max - min
        # max: 11.795407, min: -14.705297
        #   => change signs to reflect that lower is better: min: -11.795407, max: 14.705297
        # max - min = 14.705297 - -11.795407 = 26.50
        return docking_score, (-docking_score - min) / (max - min)

    @staticmethod
    def get_similarity(mol, targets):
        similarities = []
        most_similar_submission_mol = None
        max_similarity = 0.0
        for t in targets:
            similarity = TanimotoScorer.score_mols(t, mol)
            similarities.append(similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_submission_mol = t
        mean_distance = np.mean(similarities)
        return mean_distance, max_similarity, most_similar_submission_mol

    @staticmethod
    def validate_geometry(mol):
        # validate the geometry; this will raise an exception if the geometry is non-sensical
        mol2 = Chem.AddHs(mol, explicitOnly=False)
        AllChem.EmbedMolecule(mol2)
        AllChem.UFFOptimizeMolecule(mol2)

    @staticmethod
    def get_first_stereoisomer(mol):
        opts = StereoEnumerationOptions(tryEmbedding=True)
        isomers = EnumerateStereoisomers(mol, options=opts)
        return Chem.MolToSmiles(next(isomers), isomericSmiles=True)

    @staticmethod
    def log_top_best(pairs, props, top_n, logg):
        all_best = reversed(list(reversed(sorted(pairs.items(), key=lambda kv: kv[1][0])))[:top_n])
        for i, ab in enumerate(all_best):
            logg.info("%d. %s, %s; %s" % (top_n - i, ab[0], str(ab[1]), str(props[ab[0]])))

    @staticmethod
    def log_top_best_with_descriptors(pairs, props, top_n, targets, logg):
        all_best = reversed(list(reversed(sorted(pairs.items(), key=lambda kv: kv[1][0])))[:top_n])
        for i, ab in enumerate(all_best):
            smiles = ab[0]
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            qed = Utils.get_qed(mol)
            sa = Utils.get_sa_normalized(mol)
            mean_distance, max_similarity, most_similar_submission_mol = Utils.get_similarity(mol, targets)
            most_similar = Chem.MolToSmiles(most_similar_submission_mol)
            logg.info("%d. %s, %s; %s; QED: %s, SA norm.: %s, mean sim.: %s, max sim.: %s, most sim.: %s" %
                      (top_n - i, ab[0], str(ab[1]), props[ab[0]], qed, sa, mean_distance, max_similarity, most_similar))