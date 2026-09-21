"""
Data types for time-resolved trajectory data.

A trajectory is more than a stack of samples: consecutive frames are separated
by a fixed amount of physical time, the ``frame_stride`` (e.g. 0.1 ps).
Dynamical metrics (VAMP scores, implied timescales, TICA) are only comparable
between two sets of trajectories if that stride is known, because a lag
expressed in *frames* corresponds to different *physical* times whenever two
sets of trajectories were saved at different strides.

Therefore trajectories are represented by :class:`Trajectory` (frames plus the
time between them) and :class:`TrajectoryEnsemble` (several trajectories that
share dimension and frame stride) instead of a plain ``np.ndarray``.
"""

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


def _format_time(value: float, time_unit: str | None) -> str:
    return f"{value:g}" if time_unit is None else f"{value:g} {time_unit}"


@dataclass
class Trajectory:
    """
    A single time-ordered trajectory.

    Attributes
    ----------
    frames : np.ndarray
        Shape (n_frames, D). Frames in time order.
        Molecular frames of shape (n_frames, n_atoms, 3) are automatically
        flattened to (n_frames, n_atoms * 3).

    frame_stride : float
        Physical time between two consecutive frames, i.e. the save interval of
        the simulation (integration step size times save stride). Must be > 0.

    time_unit : str | None
        Optional name of the time unit (e.g. "ps"). This is a label used for
        error messages and metric keys only; no unit conversion is ever
        performed. Making sure that reference and predicted trajectories are
        expressed in the same unit is the responsibility of the caller.
    """

    frames: np.ndarray
    frame_stride: float
    time_unit: str | None = None

    def __post_init__(self):
        frames = np.asarray(self.frames)

        # Flatten molecular frames: (n_frames, n_atoms, 3) -> (n_frames, n_atoms * 3)
        if frames.ndim == 3:
            # Be explicit (not use -1) to also handle n_frames = 0
            n_frames, n_atoms, n_spatial = frames.shape
            frames = frames.reshape((n_frames, n_atoms * n_spatial))

        if frames.ndim != 2:
            raise ValueError(
                f"Trajectory frames must have shape (n_frames, dim) or "
                f"(n_frames, n_atoms, 3), got {frames.shape}"
            )

        frame_stride = float(self.frame_stride)
        if not np.isfinite(frame_stride) or frame_stride <= 0.0:
            raise ValueError(
                f"frame_stride must be finite and > 0, got {self.frame_stride}"
            )

        self.frames = frames
        self.frame_stride = frame_stride

    def __repr__(self):
        return (
            f"Trajectory(n_frames={self.n_frames}, dim={self.dim}, "
            f"frame_stride={_format_time(self.frame_stride, self.time_unit)})"
        )

    def __len__(self) -> int:
        return self.n_frames

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return self.frames.shape[0]

    @property
    def dim(self) -> int:
        """Feature dimension of a single frame."""
        return self.frames.shape[1]

    @property
    def duration(self) -> float:
        """Physical time spanned between the first and the last frame."""
        return max(self.n_frames - 1, 0) * self.frame_stride

    @property
    def times(self) -> np.ndarray:
        """Shape (n_frames,). Physical time stamp of every frame."""
        return np.arange(self.n_frames) * self.frame_stride

    def to_frame_lag(self, lag_time: float, *, rtol: float = 1e-6) -> int:
        """
        Convert a physical lag time into a number of frames.

        Parameters
        ----------
        lag_time : float
            Lag in physical time units (same unit as ``frame_stride``).
        rtol : float
            Relative tolerance used to decide whether ``lag_time`` is an
            integer multiple of ``frame_stride``.

        Returns
        -------
        int
            Lag expressed in frames.
        """
        return _to_frame_lag(
            lag_time=lag_time,
            frame_stride=self.frame_stride,
            time_unit=self.time_unit,
            rtol=rtol,
        )

    def to_lag_time(self, frame_lag: int) -> float:
        """Convert a lag given in frames into a physical lag time."""
        return frame_lag * self.frame_stride

    def subsample(self, stride: int) -> "Trajectory":
        """
        Keep only every ``stride``-th frame; the frame stride grows accordingly.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        return Trajectory(
            frames=self.frames[::stride],
            frame_stride=self.frame_stride * stride,
            time_unit=self.time_unit,
        )


@dataclass
class TrajectoryEnsemble:
    """
    A set of trajectories that share feature dimension and frame stride.

    This is the trajectory counterpart of a sample batch: dynamical metrics are
    estimated over all trajectories jointly, which requires the individual
    trajectories to be commensurable in time.

    Attributes
    ----------
    trajectories : list[Trajectory]
        The individual trajectories. Lengths may differ, dimension and
        ``frame_stride`` may not.
    """

    trajectories: list[Trajectory]

    def __post_init__(self):
        trajectories = self.trajectories

        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]

        if not isinstance(trajectories, Sequence):
            raise TypeError(
                f"trajectories must be a sequence of Trajectory objects, got "
                f"'{type(trajectories).__name__}'. Use TrajectoryEnsemble.from_array("
                f"array, frame_stride=...) to build an ensemble from raw arrays."
            )

        trajectories = list(trajectories)

        invalid = [
            f"index {i}: {type(t).__name__}"
            for i, t in enumerate(trajectories)
            if not isinstance(t, Trajectory)
        ]
        if invalid:
            raise TypeError(
                f"All entries must be Trajectory objects, but the following are "
                f"not: {invalid}. Use TrajectoryEnsemble.from_array(array, "
                f"frame_stride=...) to build an ensemble from raw arrays."
            )

        if len(trajectories) == 0:
            raise ValueError("TrajectoryEnsemble must contain at least one trajectory")

        dims = {t.dim for t in trajectories}
        if len(dims) != 1:
            raise ValueError(
                f"All trajectories must have the same dimension, found {sorted(dims)}"
            )

        frame_strides = np.array([t.frame_stride for t in trajectories])
        if not np.allclose(frame_strides, frame_strides[0]):
            raise ValueError(
                f"All trajectories of an ensemble must share the same frame_stride, "
                f"found {sorted(set(frame_strides.tolist()))}. Use subsample() to "
                f"bring them onto a common stride, or build separate ensembles."
            )

        time_units = {t.time_unit for t in trajectories}
        if len(time_units) != 1:
            raise ValueError(
                f"All trajectories of an ensemble must state the same time unit "
                f"(no unit conversion is performed), found {time_units}"
            )

        self.trajectories = trajectories

    def __repr__(self):
        return (
            f"TrajectoryEnsemble(n_trajectories={self.n_trajectories}, "
            f"n_frames={self.n_frames}, dim={self.dim}, "
            f"frame_stride={_format_time(self.frame_stride, self.time_unit)})"
        )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __getitem__(self, idx) -> Trajectory:
        return self.trajectories[idx]

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories in the ensemble."""
        return len(self.trajectories)

    @property
    def n_frames(self) -> list[int]:
        """Number of frames of every individual trajectory."""
        return [t.n_frames for t in self.trajectories]

    @property
    def n_total_frames(self) -> int:
        """Total number of frames across all trajectories."""
        return int(sum(self.n_frames))

    @property
    def dim(self) -> int:
        """Feature dimension of a single frame (shared by all trajectories)."""
        return self.trajectories[0].dim

    @property
    def frame_stride(self) -> float:
        """Physical time between two consecutive frames (shared by all trajectories)."""
        return self.trajectories[0].frame_stride

    @property
    def time_unit(self) -> str | None:
        """
        Optional name of the time unit, taken from the first trajectory.

        Purely a label: no unit conversion happens anywhere, and keeping units
        consistent across ensembles is the responsibility of the caller.
        """
        return self.trajectories[0].time_unit

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | Sequence[np.ndarray],
        frame_stride: float,
        time_unit: str | None = None,
    ) -> "TrajectoryEnsemble":
        """
        Build an ensemble from raw arrays that all share one frame stride.

        Parameters
        ----------
        array : np.ndarray | Sequence[np.ndarray]
            Either one array of shape (n_trajs, n_frames, D) /
            (n_trajs, n_frames, n_atoms, 3), or a sequence of per-trajectory
            arrays of shape (n_frames, D) / (n_frames, n_atoms, 3). A sequence
            is the only way to represent trajectories of differing length.
        frame_stride : float
            Physical time between two consecutive frames.
        time_unit : str | None
            Optional name of the time unit (e.g. "ps").
        """
        if isinstance(array, np.ndarray) and array.ndim not in (3, 4):
            raise ValueError(
                f"A single array must have shape (n_trajs, n_frames, dim) or "
                f"(n_trajs, n_frames, n_atoms, 3), got {array.shape}. Pass a "
                f"sequence of arrays for a single or for ragged trajectories."
            )
        trajectory_arrays = list(array)

        return cls(
            [
                Trajectory(
                    frames=frames, frame_stride=frame_stride, time_unit=time_unit
                )
                for frames in trajectory_arrays
            ]
        )

    def frame_arrays(self) -> list[np.ndarray]:
        """
        The raw frames of every trajectory as a list of (n_frames, D) arrays.

        This is the format expected by estimators such as those of deeptime.
        """
        return [t.frames for t in self.trajectories]

    def as_array(self) -> np.ndarray:
        """
        Shape (n_trajs, n_frames, D). Requires all trajectories to be equally long.
        """
        lengths = set(self.n_frames)
        if len(lengths) != 1:
            raise ValueError(
                f"Cannot stack trajectories of differing lengths {sorted(lengths)} "
                f"into one array; use frame_arrays() instead."
            )
        return np.stack(self.frame_arrays())

    def as_samples(self) -> np.ndarray:
        """
        Shape (n_total_frames, D). All frames pooled into one sample batch.

        Time information is discarded, which is exactly what static
        (equilibrium) evaluations such as marginal histograms expect.
        """
        return np.concatenate(self.frame_arrays(), axis=0)

    def to_frame_lag(self, lag_time: float, *, rtol: float = 1e-6) -> int:
        """
        Convert a physical lag time into a number of frames.

        Parameters
        ----------
        lag_time : float
            Lag in physical time units (same unit as ``frame_stride``).
        rtol : float
            Relative tolerance used to decide whether ``lag_time`` is an
            integer multiple of ``frame_stride``.

        Returns
        -------
        int
            Lag expressed in frames.
        """
        return _to_frame_lag(
            lag_time=lag_time,
            frame_stride=self.frame_stride,
            time_unit=self.time_unit,
            rtol=rtol,
        )

    def to_lag_time(self, frame_lag: int) -> float:
        """Convert a lag given in frames into a physical lag time."""
        return frame_lag * self.frame_stride

    def subsample(self, stride: int) -> "TrajectoryEnsemble":
        """
        Keep only every ``stride``-th frame of every trajectory; the frame
        stride grows accordingly.
        """
        return TrajectoryEnsemble([t.subsample(stride) for t in self.trajectories])


def _to_frame_lag(
    lag_time: float, frame_stride: float, time_unit: str | None, rtol: float
) -> int:
    if not np.isfinite(lag_time) or lag_time <= 0.0:
        raise ValueError(f"lag_time must be finite and > 0, got {lag_time}")

    n_frames_exact = lag_time / frame_stride
    n_frames = int(round(n_frames_exact))

    if n_frames < 1:
        raise ValueError(
            f"lag_time={_format_time(lag_time, time_unit)} is shorter than the time "
            f"between two frames ({_format_time(frame_stride, time_unit)}); the "
            f"trajectories are not resolved finely enough for this lag."
        )

    if abs(n_frames_exact - n_frames) > rtol * n_frames_exact:
        raise ValueError(
            f"lag_time={_format_time(lag_time, time_unit)} is not an integer multiple "
            f"of the time between two frames "
            f"({_format_time(frame_stride, time_unit)}): it corresponds to "
            f"{n_frames_exact:g} frames."
        )

    return n_frames


def resolve_frame_lag(
    trajectories: TrajectoryEnsemble,
    lag_time: float,
    *,
    name: str = "trajectories",
    rtol: float = 1e-6,
) -> int:
    """
    Translate a physical lag time into a frame lag for one ensemble.

    This is the bridge between the physical lag time a metric is parameterized
    with and the frame lag its estimator actually consumes. It fails with a
    readable message when the requested lag is not an integer multiple of the
    ensemble's ``frame_stride``, or when it does not fit into any trajectory.

    Parameters
    ----------
    trajectories : TrajectoryEnsemble
        Ensemble the lag should be expressed in.
    lag_time : float
        Lag in physical time units (same unit as the ensemble's frame stride).
    name : str
        Name of the ensemble, used in error messages.
    rtol : float
        Relative tolerance for the "integer multiple of frame_stride" check.

    Returns
    -------
    int
        The lag in frames.
    """
    try:
        frame_lag = trajectories.to_frame_lag(lag_time, rtol=rtol)
    except ValueError as e:
        raise ValueError(f"Invalid lag time for '{name}': {e}") from e

    longest = max(trajectories.n_frames)
    if frame_lag >= longest:
        unit = trajectories.time_unit
        longest_duration = trajectories.to_lag_time(max(longest - 1, 0))
        raise ValueError(
            f"A lag of {frame_lag} frames ({_format_time(lag_time, unit)}) does not "
            f"fit into any trajectory of '{name}' (longest trajectory: {longest} "
            f"frames, {_format_time(longest_duration, unit)})."
        )

    return frame_lag
