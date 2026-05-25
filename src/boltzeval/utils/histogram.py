from typing import Any
import numpy as np


class Histogram:
    def __init__(
        self,
        counts: np.ndarray,
        bin_edges: tuple[np.ndarray, ...],
        n_producing_samples: int,
    ):
        """
        Parameters
        ----------
        counts : np.ndarray
            Histogram bin counts.

            Shape:
                (n0, n1, ..., nD-1)

            Example:
                1D -> (N,)
                2D -> (Nx, Ny)

        bin_edges : np.ndarray | tuple[np.ndarray, ...]
            Bin boundaries for each histogram dimension.

            Shapes:
                1D -> ((N+1,),)
                2D -> ((Nx+1,), (Ny+1,))

            Constraint:
                len(bin_edges) == counts.ndim

        n_producing_samples : int
            Total number of samples used to build the histogram.
        """
        counts = np.asarray(counts, dtype=float)

        if isinstance(bin_edges, np.ndarray):
            # 1D for convenience
            bin_edges = (bin_edges,)

        if counts.sum() <= 0:
            raise ValueError("counts must sum to a positive value")

        if counts.ndim != len(bin_edges):
            raise ValueError("counts.ndim must equal len(bin_edges)")

        expected_shape = tuple(len(edges) - 1 for edges in bin_edges)

        if counts.shape != expected_shape:
            raise ValueError(
                f"counts.shape={counts.shape} " f"!= expected_shape={expected_shape}"
            )

        # check equal bin spacing in every dimension
        for axis, edges in enumerate(bin_edges):
            widths = np.diff(edges)

            if not np.allclose(widths, widths[0]):
                raise ValueError(f"bin_edges[{axis}] are not uniformly spaced")

        self._normalized_counts: np.ndarray = counts / counts.sum()

        self._bin_edges = tuple(np.asarray(edges, dtype=float) for edges in bin_edges)

        self._n_producing_samples = int(n_producing_samples)

    def __repr__(self):
        mean_str = ",".join(f"{x:.2f}" for x in self.get_mean())

        std_str = ",".join(f"{x:.2f}" for x in self.get_std())

        return f"Histogram{self.ndim}D(" f"mean=({mean_str})," f"std=({std_str})" f")"

    @property
    def ndim(self) -> int:
        """
        Number of histogram dimensions.

        Returns
        -------
        int
        """
        return self._normalized_counts.ndim

    @property
    def n_producing_samples(self) -> int:
        """
        Total number of samples used to construct the histogram.

        Returns
        -------
        int
        """
        return self._n_producing_samples

    @property
    def bin_edges(
        self,
    ) -> tuple[np.ndarray, ...]:
        """
        Bin boundaries for each histogram dimension.

        Returns
        -------
        tuple[np.ndarray, ...]

            Example:
                1D -> ((N+1,),)
                2D -> ((Nx+1,), (Ny+1,))
        """
        return self._bin_edges

    def get_num_bins(self) -> tuple[int, ...]:
        """
        Number of bins along each histogram dimension.

        Returns
        -------
        tuple[int, ...]

            Example:
                1D -> (N,)
                2D -> (Nx, Ny)
                ND -> (n0, n1, ..., nD-1)
        """
        return self._normalized_counts.shape

    def get_normalized_counts(self) -> np.ndarray:
        """
        Empirical probability mass per histogram bin.

        Each bin value represents the fraction of total samples that fall into that bin:
            value = (samples in bin) / (total samples)

        Returns
        -------
        np.ndarray

            Shape:
                (n0, n1, ..., nD-1)

            Example:
                1D -> (N,)
                2D -> (Nx, Ny)

            All values sum to 1.
        """
        counts = self._normalized_counts
        return counts / counts.sum()

    def get_approximate_absolute_counts(
        self,
    ) -> np.ndarray:
        """
        Empirical sample counts per histogram bin (approximate up to rounding errors).

        Each bin value estimates:
            number of samples in that bin

        Computed as:
            (bin probability mass) x (total samples)

        Returns
        -------
        np.ndarray

            Shape:
                (n0, n1, ..., nD-1)

            Example:
                1D -> (N,)
                2D -> (Nx, Ny)

            Sum ≈ n_producing_samples.
        """
        return self.get_normalized_counts() * self.n_producing_samples

    def get_as_density(self) -> np.ndarray:
        """
        Probability density per histogram bin.

        Each bin value is:
            (fraction of samples in bin) / (bin volume)

        Returns
        -------
        np.ndarray

            Shape:
                (n0, n1, ..., nD-1)

            Example:
                1D -> (N,)
                2D -> (Nx, Ny)
        """
        return self.get_normalized_counts() / self.get_bin_volume()

    def get_support_range(
        self,
    ) -> tuple[tuple[float, float], ...]:
        """
        Data range covered by the histogram.

        Returns
        -------
        tuple[tuple[float, float], ...]

            One entry per histogram dimension.

            Example:
                1D -> ((xmin, xmax),)
                2D -> (
                    (xmin, xmax),
                    (ymin, ymax),
                )
        """
        return tuple((edges[0], edges[-1]) for edges in self.bin_edges)

    def get_bin_centers(
        self,
    ) -> tuple[np.ndarray, ...]:
        """
        Center coordinate of each histogram bin.

        Returns
        -------
        tuple[np.ndarray, ...]

            One array per histogram dimension.

            Shapes:
                1D -> ((N,),)
                2D -> ((Nx,), (Ny,))
        """
        return tuple(0.5 * (edges[:-1] + edges[1:]) for edges in self.bin_edges)

    def get_bin_volume(self) -> float:
        """
        Volume of a single histogram bin.

        Interpretation:
            1D -> bin width
            2D -> bin area
            ND -> hypervolume

        Returns
        -------
        float
        """
        volume = 1.0

        for edges, n_bins in zip(
            self.bin_edges,
            self.get_num_bins(),
        ):
            volume *= (edges[-1] - edges[0]) / n_bins

        return float(volume)

    def get_mean(self) -> np.ndarray:
        """
        Mean coordinate along each histogram dimension.

        Returns
        -------
        np.ndarray

            Shape:
                (D,)

            Example:
                1D -> (1,)
                2D -> (2,)
        """
        counts = self.get_normalized_counts()

        means = []

        for axis, centers in enumerate(self.get_bin_centers()):
            reduction_axes = tuple(i for i in range(self.ndim) if i != axis)
            marginal: np.ndarray = counts.sum(axis=reduction_axes)

            means.append((marginal * centers).sum())

        return np.asarray(means)

    def get_std(self) -> np.ndarray:
        """
        Standard deviation along each histogram dimension.

        Returns
        -------
        np.ndarray

            Shape:
                (D,)

            Example:
                1D -> (1,)
                2D -> (2,)
        """
        counts = self.get_normalized_counts()
        means = self.get_mean()

        stds = []

        for axis, centers in enumerate(self.get_bin_centers()):
            reduction_axes = tuple(i for i in range(self.ndim) if i != axis)
            marginal: np.ndarray = counts.sum(axis=reduction_axes)

            var: float = (marginal * (centers - means[axis]) ** 2).sum()

            stds.append(np.sqrt(var))

        return np.asarray(stds)

    def get_state(
        self,
    ) -> dict[str, np.ndarray | int | Any]:
        """
        Serializable histogram state.

        Contains all information required to reconstruct the histogram,
        including:
            - bin counts
            - bin boundaries
            - number of samples

        Returns
        -------
        dict
        """
        return {
            "counts": self._normalized_counts,
            "bin_edges": self.bin_edges,
            "n_producing_samples": (self.n_producing_samples),
        }

    def save(self, fpath: str):
        """
        Save histogram to .npz file.
        """
        state = self.get_state()

        savez_dict = {
            "counts": state["counts"],
            "n_producing_samples": state["n_producing_samples"],
        }

        for i, edges in enumerate(state["bin_edges"]):
            savez_dict[f"bin_edges_{i}"] = edges

        np.savez(
            fpath,
            allow_pickle=False,
            **savez_dict,
        )

    @classmethod
    def load(cls, fpath: str) -> "Histogram":
        """
        Load histogram from .npz file.

        Returns
        -------
        Histogram
        """
        data = np.load(fpath)

        bin_edges = tuple(
            data[key] for key in sorted(data.files) if key.startswith("bin_edges_")
        )

        return cls(
            counts=data["counts"],
            bin_edges=bin_edges,
            n_producing_samples=int(data["n_producing_samples"]),
        )

    @classmethod
    def from_samples(
        cls,
        data: np.ndarray,
        bins: int | tuple[int, ...] = 100,
        data_range: (
            tuple[tuple[float, float], ...]
            | tuple[float, float]
            | tuple[float, float, float, float]
            | None
        ) = None,
    ) -> "Histogram":
        """
        Create an N-dimensional histogram from raw samples.

        Parameters
        ----------
        data : np.ndarray
            Input samples.

            Shape:
                (n_samples, D)

            Each row is one sample in D dimensions.

        bins : int | tuple[int, ...]
            Number of bins per dimension.

            - int -> same number of bins in all dimensions
            - tuple -> one entry per dimension

        data_range : tuple[tuple[float, float], ...]
                    | tuple[float, float]
                    | tuple[float, float, float, float]
                    | None
            Defines the binning range for each dimension.

            Supported formats:

            1. Full per-dimension specification (recommended)
                ((min0, max0), (min1, max1), ..., (minD, maxD))

                - Explicitly defines min/max for each dimension.
                - Length must match `data.shape[1]`.

            2. Single 2-tuple (1D convenience form)
                (min, max)

                - Interpreted as a single dimension range applied equally for all dimensions.
                - Equivalent to: ((min, max),)*n_dims

            3. Flat 4-tuple (2D convenience form)
                (min_x, max_x, min_y, max_y)

                - Interpreted as:
                    ((min_x, max_x), (min_y, max_y))

                - Only valid when data is 2-dimensional.

            4. None (default)
                - Ranges are inferred from the data:
                    min/max computed per dimension via:
                        data.min(axis=0), data.max(axis=0)

                - Useful for quick exploratory histograms but may lead to
                  inconsistent binning across datasets.

        Returns
        -------
        Histogram
            ND histogram instance.
        """
        data = np.asarray(data)

        if data.ndim == 1:
            data = np.expand_dims(data, -1)

        if data.ndim != 2:
            raise ValueError("data must have shape (n_samples, n_dims) or (n_samples,)")

        n_samples, n_dims = data.shape

        if isinstance(bins, int):
            bins = (bins,) * n_dims

        if len(bins) != n_dims:
            raise ValueError("bins must match number of dimensions")

        if data_range is None:
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            data_range = tuple(zip(mins, maxs))

        if isinstance(data_range, tuple) and isinstance(data_range[0], float):
            if len(data_range) == 2:
                data_range = (data_range,) * n_dims
            elif len(data_range) == 4:
                min_x, max_x, min_y, max_y = data_range
                data_range = ((min_x, max_x), (min_y, max_y))
            else:
                raise RuntimeError(
                    f"Unexpected data_range object of length {len(data_range)}"
                )

        if len(data_range) != n_dims:
            raise ValueError(
                f"data_range must match number of dimensions ({len(data_range)}!={n_dims})"
            )

        hist, edges = np.histogramdd(
            data,
            bins=bins,
            range=data_range,
        )

        bin_edges = tuple(edges)

        return cls(
            counts=hist,
            bin_edges=bin_edges,
            n_producing_samples=n_samples,
        )
