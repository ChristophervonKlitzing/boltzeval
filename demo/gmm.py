"""
The toy system used by the boltzeval demos.

A two-dimensional Gaussian mixture with two modes plays the role of the
Boltzmann distribution we want to sample: it is cheap, it has a tractable
density and gradient, and its two well-separated modes make both static
(marginals, free energies) and dynamical (VAMP, timescales) metrics
interesting - the slow process is the hopping between the two modes.

Reference data is produced by exact sampling (for the sample demo) and by
overdamped Langevin dynamics (for the trajectory demo). The "model" being
evaluated is faked throughout by adding a little Gaussian noise to reference
data, which is enough to make the metrics show a small but visible gap.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from boltzeval.utils.trajectory import TrajectoryEnsemble


@dataclass
class GaussianMixture2D:
    """
    An isotropic two-dimensional Gaussian mixture.

    Attributes
    ----------
    means : np.ndarray
        Shape (n_modes, 2). Mode centers.
    std : float
        Standard deviation shared by all modes.
    weights : np.ndarray
        Shape (n_modes,). Mixture weights, summing to one.
    """

    means: np.ndarray
    std: float = 0.5
    weights: np.ndarray = None

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=float)

        if self.weights is None:
            self.weights = np.full(self.n_modes, 1.0 / self.n_modes)

        self.weights = np.asarray(self.weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

    @property
    def n_modes(self) -> int:
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def _component_log_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Shape (batch, n_modes). log[w_k * N(x | mu_k, std^2 I)] per component.
        """
        # (batch, n_modes, dim)
        deltas = x[:, None, :] - self.means[None, :, :]
        sq_dists = np.sum(deltas**2, axis=-1)

        normalizer = self.dim * np.log(2.0 * np.pi * self.std**2)
        return np.log(self.weights)[None, :] - 0.5 * (
            sq_dists / self.std**2 + normalizer
        )

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Shape (batch,). Normalized log density of the mixture.
        """
        return logsumexp(self._component_log_probs(np.asarray(x)), axis=-1)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Shape (batch, dim). Gradient of log_prob, i.e. the negative force of the
        potential U(x) = -log p(x) (in units of kT).
        """
        x = np.asarray(x)

        # Responsibilities: softmax over the component log probabilities
        log_component = self._component_log_probs(x)
        responsibilities = np.exp(
            log_component - logsumexp(log_component, axis=-1, keepdims=True)
        )

        # (batch, n_modes, dim)
        deltas = self.means[None, :, :] - x[:, None, :]
        return np.sum(responsibilities[:, :, None] * deltas, axis=1) / self.std**2

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Shape (n_samples, dim). Exact i.i.d. samples from the mixture.
        """
        modes = rng.choice(self.n_modes, size=n_samples, p=self.weights)
        return self.means[modes] + self.std * rng.normal(size=(n_samples, self.dim))


def make_double_well_mixture() -> GaussianMixture2D:
    """
    The two-mode mixture used by both demos.

    The weights are deliberately unequal (60/40), so that the two modes differ
    in free energy and a model that gets the mode populations wrong is
    penalized by the histogram metrics.

    The width is chosen such that the barrier between the modes is a few kT:
    high enough for mode hopping to be the slow process of the Langevin
    dynamics below, low enough for short trajectories to actually cross it and
    reach the correct mode populations.
    """
    return GaussianMixture2D(
        means=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        std=0.8,
        weights=np.array([0.6, 0.4]),
    )


def simulate_overdamped_langevin(
    model: GaussianMixture2D,
    rng: np.random.Generator,
    n_trajectories: int = 8,
    n_steps: int = 20_000,
    time_step: float = 0.005,
    save_stride: int = 1,
    time_unit: str | None = "ps",
) -> TrajectoryEnsemble:
    """
    Sample trajectories with overdamped Langevin dynamics (unit friction, kT=1):

        x_{t+1} = x_t + dt * grad log p(x_t) + sqrt(2 dt) * xi,   xi ~ N(0, I)

    whose stationary distribution is the mixture itself.

    Parameters
    ----------
    model : GaussianMixture2D
        Target distribution providing the drift.
    rng : np.random.Generator
        Source of randomness.
    n_trajectories : int
        Number of independent trajectories to run in parallel.
    n_steps : int
        Number of integration steps per trajectory.
    time_step : float
        Integration step size dt, in ``time_unit``.
    save_stride : int
        Save only every ``save_stride``-th step. The resulting trajectories
        have a frame stride of ``time_step * save_stride``, which is exactly
        the information a bare array of frames would lose.
    time_unit : str | None
        Name of the time unit, carried along as a label.

    Returns
    -------
    TrajectoryEnsemble
        ``n_trajectories`` trajectories of ``n_steps // save_stride`` frames.
    """
    # Start from the equilibrium distribution, so no burn-in is needed
    x = model.sample(n_trajectories, rng)

    saved_frames = []
    noise_scale = np.sqrt(2.0 * time_step)

    for step in range(n_steps):
        noise = rng.normal(size=x.shape)
        x = x + time_step * model.grad_log_prob(x) + noise_scale * noise

        if (step + 1) % save_stride == 0:
            saved_frames.append(x.copy())

    # (n_saved_frames, n_trajectories, dim) -> (n_trajectories, n_saved_frames, dim)
    frames = np.stack(saved_frames, axis=0).transpose(1, 0, 2)

    return TrajectoryEnsemble.from_array(
        frames,
        frame_stride=time_step * save_stride,
        time_unit=time_unit,
    )


def add_gaussian_noise(
    samples: np.ndarray, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Fake a model by blurring reference samples with Gaussian noise.

    Blurring an isotropic mixture with N(0, noise_std^2) yields the same
    mixture with a slightly larger width, so the faked model is a genuinely
    (but only mildly) wrong distribution - just what a demo needs.
    """
    return samples + noise_std * rng.normal(size=samples.shape)


def add_gaussian_noise_to_trajectories(
    trajectories: TrajectoryEnsemble, noise_std: float, rng: np.random.Generator
) -> TrajectoryEnsemble:
    """
    Same faking as :func:`add_gaussian_noise`, applied frame-wise.

    The frame stride is carried over unchanged: perturbing the coordinates does
    not change how much time passes between two frames.
    """
    return TrajectoryEnsemble.from_array(
        [add_gaussian_noise(traj.frames, noise_std, rng) for traj in trajectories],
        frame_stride=trajectories.frame_stride,
        time_unit=trajectories.time_unit,
    )
