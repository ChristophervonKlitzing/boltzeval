from typing import Protocol

import numpy as np
import mdtraj as md


class FeatureTransform(Protocol):
    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        Converts samples of shape (batch, n_atoms, 3) or (batch, d) into features.
        """
        pass


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

    pdb = app.PDBFile("test_files/aldp_topology.pdb")
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
