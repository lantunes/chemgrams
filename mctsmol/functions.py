import numpy as np
from rdkit.Chem import AllChem


def simple_torsional_potential(angles):
    """
    :param angles: a list of angles in degrees 
    :return: the total torsional potential
    """
    sum = 0.0
    for angle in angles:
        sum += 1 + np.cos(np.deg2rad(angle)) + np.cos(np.deg2rad(3 * angle))
    return sum


def uff_potential(mol):
    """
    Accepts an rdkit Mol, optimizes the geometry, and returns the energy.
    :param mol: an rdkit Mol
    :return: the energy
    """
    AllChem.UFFOptimizeMolecule(mol)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    energy = ff.CalcEnergy()
    return energy


def mmff94_potential(mol):
    """
    Accepts an rdkit Mol, optimizes the geometry, and returns the energy.
    :param mol: an rdkit Mol
    :return: the energy
    """
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    opt_fail = ff.Minimize(maxIts=1000,forceTol=0.0001,energyTol=1e-06)
    energy = ff.CalcEnergy()
    return energy

