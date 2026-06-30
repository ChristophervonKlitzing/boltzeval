import unittest
import tempfile
import numpy as np

from boltzeval.utils.histogram import Histogram


class TestHistogram(unittest.TestCase):
    def setUp(self):
        # 1D example
        self.counts_1d = np.array([1, 2, 3])
        self.edges_1d = (np.array([0.0, 1.0, 2.0, 3.0]),)

        self.hist1d = Histogram(
            counts=self.counts_1d,
            bin_edges=self.edges_1d,
            n_producing_samples=60,
        )

        # 2D example
        self.counts_2d = np.array(
            [
                [1, 2],
                [3, 4],
            ]
        )

        self.edges_2d = (
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 20.0, 30.0]),
        )

        self.hist2d = Histogram(
            counts=self.counts_2d,
            bin_edges=self.edges_2d,
            n_producing_samples=100,
        )

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            Histogram(
                counts=np.ones((3, 4)),
                bin_edges=(
                    np.arange(4),
                    np.arange(4),
                ),
                n_producing_samples=1,
            )

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------

    def test_ndim(self):
        self.assertEqual(self.hist1d.ndim, 1)
        self.assertEqual(self.hist2d.ndim, 2)

    def test_num_bins(self):
        self.assertEqual(
            self.hist1d.get_num_bins(),
            (3,),
        )

        self.assertEqual(
            self.hist2d.get_num_bins(),
            (2, 2),
        )

    def test_n_producing_samples(self):
        self.assertEqual(
            self.hist1d.n_producing_samples,
            60,
        )

    # -------------------------------------------------------------------------
    # Normalized counts
    # -------------------------------------------------------------------------

    def test_normalized_counts_sum_to_one(self):
        self.assertAlmostEqual(
            self.hist1d.get_normalized_counts().sum(),
            1.0,
        )

        self.assertAlmostEqual(
            self.hist2d.get_normalized_counts().sum(),
            1.0,
        )

    def test_normalized_counts_values(self):
        expected = np.array(
            [
                1 / 6,
                2 / 6,
                3 / 6,
            ]
        )

        np.testing.assert_allclose(
            self.hist1d.get_normalized_counts(),
            expected,
        )

    # -------------------------------------------------------------------------
    # Approximate counts
    # -------------------------------------------------------------------------

    def test_absolute_counts(self):
        expected = np.array(
            [
                10,
                20,
                30,
            ]
        )

        np.testing.assert_allclose(
            self.hist1d.get_approximate_absolute_counts(),
            expected,
        )

    # -------------------------------------------------------------------------
    # Bin geometry
    # -------------------------------------------------------------------------

    def test_support_range(self):
        self.assertEqual(
            self.hist1d.get_support_range(),
            ((0.0, 3.0),),
        )

        self.assertEqual(
            self.hist2d.get_support_range(),
            (
                (0.0, 2.0),
                (10.0, 30.0),
            ),
        )

    def test_bin_centers(self):
        centers = self.hist1d.get_bin_centers()

        np.testing.assert_allclose(
            centers[0],
            np.array([0.5, 1.5, 2.5]),
        )

        x, y = self.hist2d.get_bin_centers()

        np.testing.assert_allclose(
            x,
            np.array([0.5, 1.5]),
        )

        np.testing.assert_allclose(
            y,
            np.array([15.0, 25.0]),
        )

    def test_bin_volume(self):
        self.assertAlmostEqual(
            self.hist1d.get_bin_volume(),
            1.0,
        )

        self.assertAlmostEqual(
            self.hist2d.get_bin_volume(),
            10.0,
        )

    # -------------------------------------------------------------------------
    # Density
    # -------------------------------------------------------------------------

    def test_density_1D(self):
        density = self.hist1d.get_as_density()

        expected = self.hist1d.get_normalized_counts() / 1.0

        np.testing.assert_allclose(
            density,
            expected,
        )

    def test_density_2D(self):
        density = self.hist2d.get_as_density()

        expected = self.hist2d.get_normalized_counts() / 10.0

        np.testing.assert_allclose(
            density,
            expected,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def test_mean(self):
        mean = self.hist1d.get_mean()

        expected = np.array([(1 / 6) * 0.5 + (2 / 6) * 1.5 + (3 / 6) * 2.5])

        np.testing.assert_allclose(
            mean,
            expected,
        )

    def test_std(self):
        std = self.hist1d.get_std()

        self.assertEqual(std.shape, (1,))
        self.assertGreater(std[0], 0)

    # -------------------------------------------------------------------------
    # State / serialization
    # -------------------------------------------------------------------------

    def test_get_state(self):
        state = self.hist1d.get_state()

        self.assertIn("counts", state)
        self.assertIn("bin_edges", state)
        self.assertIn(
            "n_producing_samples",
            state,
        )

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:

            self.hist2d.save(tmp.name)

            loaded = Histogram.load(tmp.name)

            np.testing.assert_allclose(
                loaded.get_normalized_counts(),
                self.hist2d.get_normalized_counts(),
            )

            self.assertEqual(
                loaded.get_num_bins(),
                self.hist2d.get_num_bins(),
            )

            self.assertEqual(
                loaded.n_producing_samples,
                self.hist2d.n_producing_samples,
            )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def test_repr(self):
        r = repr(self.hist1d)

        self.assertIn("Histogram1D", r)
        self.assertIn("mean=", r)
        self.assertIn("std=", r)


if __name__ == "__main__":
    unittest.main()
