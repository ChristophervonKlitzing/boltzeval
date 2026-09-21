"""
Feature transforms turning atomic coordinates into features for TICA, VAMP and
related analyses.

Parts of the dihedral featurization (topology construction and the dihedral
selection) are adapted from the TITO code base
(https://github.com/olsson-group/tito), which is MIT licensed.
"""

from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np
import mdtraj as md

from boltzeval.utils.shape_utils import reshape_to_molecular
from boltzeval.utils.trajectory import TrajectoryEnsemble

try:
    from rdkit import Chem
except ImportError:  # rdkit is an optional dependency of boltzeval
    Chem = None

if TYPE_CHECKING:
    # Import again for type checkers, so that "Chem.Mol" annotations resolve
    # even in an environment without rdkit
    from rdkit import Chem


class FeatureTransform(Protocol):
    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        Converts samples of shape (batch, n_atoms, 3) or (batch, d) into features.
        """
        pass


def featurize_trajectories(
    trajectories: TrajectoryEnsemble,
    feature_transform: FeatureTransform,
) -> list[np.ndarray]:
    """
    Apply a feature transform to the frames of every trajectory of an ensemble.

    Returns
    -------
    list[np.ndarray]
        One (n_frames, n_features) array per trajectory. A list (rather than a
        stacked array) keeps trajectories of differing length usable and is the
        input format expected by the deeptime estimators.
    """
    return [feature_transform(traj.frames) for traj in trajectories]


def _get_distances(xyz):
    """
    Returns the flattened pair-wise distances between all atoms in a batched fashion
    """
    distance_matrix_ca: np.ndarray = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


class AtomicDistanceFeatureTransform(FeatureTransform):
    """
    Extracts the flattened pair-wise interatomic distances
    """

    def __call__(self, samples):
        if samples.ndim == 2:
            batch = samples.shape[0]
            samples = samples.reshape((batch, -1, 3))

        return _get_distances(samples)


class PhiPsiTorsionFeatureTransform(FeatureTransform):
    def __init__(self, topology: md.Topology):
        super().__init__()
        self._topology = topology

    def __call__(self, samples):
        if samples.ndim == 2:
            batch = samples.shape[0]
            samples = samples.reshape((batch, -1, 3))

        traj = md.Trajectory(xyz=samples, topology=self._topology)
        phi = md.compute_phi(traj)[1][:, 0]
        psi = md.compute_psi(traj)[1][:, 0]

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        features = np.stack([cos_phi, sin_phi, cos_psi, sin_psi]).T
        return features


class IdentityFeatureTransform(FeatureTransform):
    def __call__(self, samples):
        return samples


def _find_dihedral_atoms(mol: "Chem.Mol") -> list[tuple[int, int, int, int]]:
    """
    Pick one dihedral per bond of a molecule.

    For every bond (a1, a2) the first neighbor of a1 other than a2 and the
    first neighbor of a2 other than a1 are chosen, which yields the atom
    quadruple (n1, a1, a2, n2). Bonds that are terminal on either side cannot
    form a dihedral and are skipped.

    Parameters
    ----------
    mol : Chem.Mol
        An rdkit molecule, e.g. from ``rdkit.Chem.MolFromMolBlock``. Only its
        bond graph is used, so a molecule without conformers is enough. Its
        atom order must match the atom order of the samples to be featurized.

    Returns
    -------
    list[tuple[int, int, int, int]]
        One atom-index quadruple per non-terminal bond.
    """
    if Chem is None:
        raise ImportError(
            "rdkit is required to derive the dihedral atoms of a molecule, but "
            "is not installed. It is an optional dependency of boltzeval: "
            "install it with 'pip install rdkit'."
        )

    dihedral_atoms = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()

        neighbors1 = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(atom1).GetNeighbors()
            if atom.GetIdx() != atom2
        ]
        neighbors2 = [
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(atom2).GetNeighbors()
            if atom.GetIdx() != atom1
        ]

        if not neighbors1 or not neighbors2:
            # Terminal bond: there is no atom to span a dihedral with
            continue

        dihedral_atoms.append((neighbors1[0], atom1, atom2, neighbors2[0]))

    return dihedral_atoms


def _get_topology(atom_numbers: Sequence[int] | np.ndarray) -> md.Topology:
    """
    Build a flat topology holding one residue with the given atoms.

    Parameters
    ----------
    atom_numbers : Sequence[int] | np.ndarray
        Shape (n_atoms,). Atomic number of every atom, in the order the
        coordinates are stored in.

    Returns
    -------
    md.Topology
        A topology with a single chain and a single residue named "RES", whose
        atoms are named by element and index (e.g. "carbon0", "hydrogen1").
    """
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = md.element.Element.getByAtomicNumber(int(atom_number))
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology


def _get_mdtraj(
    samples: np.ndarray, atom_numbers: Sequence[int] | np.ndarray
) -> md.Trajectory:
    """
    Wrap a batch of structures into an mdtraj trajectory.

    Parameters
    ----------
    samples : np.ndarray
        Shape (..., n_atoms, 3). Any leading dimensions are flattened into the
        frame dimension.
    atom_numbers : Sequence[int] | np.ndarray
        Shape (n_atoms,). Atomic number of every atom.

    Returns
    -------
    md.Trajectory
        A trajectory of ``prod(leading dimensions)`` frames.
    """
    topology = _get_topology(atom_numbers)
    samples = samples.reshape(-1, *samples.shape[-2:])
    return md.Trajectory(samples, topology)


def _get_dihedrals(
    samples: np.ndarray, dihedral_atoms: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Compute dihedral angles of given atom quadruples for a batch of structures.

    Parameters
    ----------
    samples : np.ndarray
        Shape (batch, n_atoms, 3). Atomic coordinates.
    dihedral_atoms : list[tuple[int, int, int, int]]
        Atom index quadruples defining the dihedrals.

    Returns
    -------
    np.ndarray
        Shape (batch, n_dihedrals). Angles in radians, in (-pi, pi].
    """
    # The dihedrals are defined by explicit atom indices and are therefore
    # purely geometric: the elements of the topology do not enter, so a
    # placeholder of all-hydrogen atoms is enough to build the trajectory.
    traj = _get_mdtraj(samples, np.ones(samples.shape[-2]))
    return md.compute_dihedrals(traj, dihedral_atoms)


def _get_sinusoids(dihedrals: np.ndarray) -> np.ndarray:
    """
    Encode angles as their cosines and sines.

    This removes the discontinuity at +-pi, which would otherwise dominate
    distances and covariances computed on the raw angles.

    Parameters
    ----------
    dihedrals : np.ndarray
        Shape (batch, n_dihedrals). Angles in radians.

    Returns
    -------
    np.ndarray
        Shape (batch, 2 * n_dihedrals). The cosines followed by the sines.
    """
    return np.concatenate((np.cos(dihedrals), np.sin(dihedrals)), axis=-1)


class DihedralsFeatureTransform(FeatureTransform):
    """
    Encodes one dihedral angle per bond as its cosine and sine.

    The dihedrals are derived from the bond graph of an rdkit molecule: every
    non-terminal bond contributes one dihedral, spanned by the first neighbor
    on either side (see :func:`find_dihedral_atoms`). Each angle enters the
    features as a (cos, sin) pair, so the representation is continuous across
    the +-pi wrap-around.

    This is the featurization the TITO paper ("Transferable generative models
    bridge femtosecond to nanosecond time-step molecular dynamics",
    https://www.science.org/doi/10.1126/sciadv.aed2333) uses to construct its
    TICA plots and related analyses.

    Parameters
    ----------
    mol : Chem.Mol
        An rdkit molecule, e.g. from ``rdkit.Chem.MolFromMolBlock``, whose atom
        order matches the atom order of the samples to be featurized. rdkit is
        an optional dependency of boltzeval.
    """

    def __init__(self, mol: "Chem.Mol"):
        super().__init__()
        self._dihedral_atoms = _find_dihedral_atoms(mol)

    @property
    def dihedral_atoms(self) -> list[tuple[int, int, int, int]]:
        """The atom index quadruples the dihedrals are computed from."""
        return self._dihedral_atoms

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        samples : np.ndarray
            Shape (batch, n_atoms, 3) or (batch, n_atoms * 3).

        Returns
        -------
        np.ndarray
            Shape (batch, 2 * n_dihedrals).
        """
        samples = reshape_to_molecular(samples)
        return _get_sinusoids(_get_dihedrals(samples, self._dihedral_atoms))


"""
def create_C_alpha_distance_feature_transform(
    topology: md.Topology,
) -> FeatureTransform:
    def _transform(samples: np.ndarray) -> np.ndarray: ...


def create_sidechain_angles_feature_transform(
    topology: md.Topology,
) -> FeatureTransform:
    def _transform(samples: np.ndarray) -> np.ndarray: ...
"""


if __name__ == "__main__":
    from openmm import app

    pdb = app.PDBFile("demo/test_files/aldp_topology.pdb")
    topology = md.Topology.from_openmm(pdb.topology)
    pos = pdb.getPositions(asNumpy=True)
    print(np.linalg.norm(pos[0] - pos[2]))
    samples = np.expand_dims(pos, 0)

    phi_psi_transform = PhiPsiTorsionFeatureTransform(topology)
    features = phi_psi_transform(samples)
    print(features.shape)
    print(features)

    distance_transform = AtomicDistanceFeatureTransform()
    features = distance_transform(samples)
    print(features.shape)
    print(features)
