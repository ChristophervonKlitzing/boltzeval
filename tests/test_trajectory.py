import unittest

import numpy as np

from boltzeval.pipeline import EvalData
from boltzeval.utils.trajectory import (
    Trajectory,
    TrajectoryEnsemble,
    resolve_frame_lag,
)


class TestTrajectory(unittest.TestCase):
    def test_basic_properties(self):
        traj = Trajectory(np.zeros((10, 3)), frame_stride=0.5, time_unit="ps")

        self.assertEqual(traj.n_frames, 10)
        self.assertEqual(traj.dim, 3)
        self.assertEqual(len(traj), 10)
        self.assertAlmostEqual(traj.duration, 4.5)
        np.testing.assert_allclose(traj.times, np.arange(10) * 0.5)

    def test_molecular_frames_are_flattened(self):
        traj = Trajectory(np.zeros((10, 4, 3)), frame_stride=1.0)
        self.assertEqual(traj.frames.shape, (10, 12))

    def test_invalid_frame_stride(self):
        for invalid in [0.0, -1.0, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                Trajectory(np.zeros((10, 2)), frame_stride=invalid)

    def test_subsample_grows_frame_stride(self):
        traj = Trajectory(np.arange(10).reshape(10, 1), frame_stride=0.5)
        sub = traj.subsample(2)

        self.assertEqual(sub.n_frames, 5)
        self.assertAlmostEqual(sub.frame_stride, 1.0)
        np.testing.assert_allclose(sub.frames.ravel(), [0, 2, 4, 6, 8])


class TestTrajectoryEnsemble(unittest.TestCase):
    def setUp(self):
        self.ensemble = TrajectoryEnsemble.from_array(
            np.zeros((3, 100, 2)), frame_stride=0.2, time_unit="ps"
        )

    def test_from_array(self):
        self.assertEqual(self.ensemble.n_trajectories, 3)
        self.assertEqual(self.ensemble.n_frames, [100, 100, 100])
        self.assertEqual(self.ensemble.n_total_frames, 300)
        self.assertEqual(self.ensemble.dim, 2)
        self.assertAlmostEqual(self.ensemble.frame_stride, 0.2)
        self.assertEqual(self.ensemble.time_unit, "ps")

    def test_from_ragged_sequence(self):
        ensemble = TrajectoryEnsemble.from_array(
            [np.zeros((10, 2)), np.zeros((20, 2))], frame_stride=1.0
        )

        self.assertEqual(ensemble.n_frames, [10, 20])
        self.assertEqual(ensemble.as_samples().shape, (30, 2))
        with self.assertRaises(ValueError):
            ensemble.as_array()

    def test_as_array_and_as_samples(self):
        self.assertEqual(self.ensemble.as_array().shape, (3, 100, 2))
        self.assertEqual(self.ensemble.as_samples().shape, (300, 2))

    def test_raw_array_is_rejected(self):
        with self.assertRaises(TypeError):
            TrajectoryEnsemble(np.zeros((3, 100, 2)))

    def test_mixed_frame_strides_are_rejected(self):
        with self.assertRaises(ValueError):
            TrajectoryEnsemble(
                [
                    Trajectory(np.zeros((10, 2)), frame_stride=1.0),
                    Trajectory(np.zeros((10, 2)), frame_stride=2.0),
                ]
            )

    def test_mixed_time_units_are_rejected(self):
        with self.assertRaises(ValueError):
            TrajectoryEnsemble(
                [
                    Trajectory(np.zeros((10, 2)), frame_stride=1.0, time_unit="ps"),
                    Trajectory(np.zeros((10, 2)), frame_stride=1.0, time_unit="ns"),
                ]
            )

    def test_mixed_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            TrajectoryEnsemble(
                [
                    Trajectory(np.zeros((10, 2)), frame_stride=1.0),
                    Trajectory(np.zeros((10, 3)), frame_stride=1.0),
                ]
            )


class TestFrameLagResolution(unittest.TestCase):
    def setUp(self):
        self.ensemble = TrajectoryEnsemble.from_array(
            np.zeros((2, 100, 2)), frame_stride=0.2, time_unit="ps"
        )

    def test_lag_time_to_frames(self):
        self.assertEqual(self.ensemble.to_frame_lag(1.0), 5)
        self.assertAlmostEqual(self.ensemble.to_lag_time(5), 1.0)

    def test_lag_time_must_be_multiple_of_frame_stride(self):
        with self.assertRaises(ValueError):
            self.ensemble.to_frame_lag(0.3)

    def test_lag_time_below_frame_stride(self):
        with self.assertRaises(ValueError):
            self.ensemble.to_frame_lag(0.01)

    def test_lag_must_fit_into_trajectories(self):
        with self.assertRaises(ValueError):
            resolve_frame_lag(self.ensemble, 1000.0)


class TestEvalDataTrajectories(unittest.TestCase):
    def setUp(self):
        self.trajs_true = TrajectoryEnsemble.from_array(
            np.zeros((2, 100, 2)), frame_stride=0.2, time_unit="ps"
        )
        self.trajs_pred = TrajectoryEnsemble.from_array(
            np.zeros((2, 50, 2)), frame_stride=0.4, time_unit="ps"
        )

    def test_differing_frame_strides_are_allowed(self):
        data = EvalData(trajs_true=self.trajs_true, trajs_pred=self.trajs_pred)

        self.assertAlmostEqual(data.trajs_true.frame_stride, 0.2)
        self.assertAlmostEqual(data.trajs_pred.frame_stride, 0.4)

    def test_single_trajectory_is_wrapped(self):
        data = EvalData(trajs_true=Trajectory(np.zeros((10, 2)), frame_stride=1.0))

        self.assertIsInstance(data.trajs_true, TrajectoryEnsemble)
        self.assertEqual(data.trajs_true.n_trajectories, 1)

    def test_raw_array_is_rejected(self):
        with self.assertRaises(TypeError):
            EvalData(trajs_true=np.zeros((2, 100, 2)))

    def test_time_units_must_match(self):
        trajs_pred_ns = TrajectoryEnsemble.from_array(
            np.zeros((2, 50, 2)), frame_stride=0.4, time_unit="ns"
        )
        with self.assertRaises(ValueError):
            EvalData(trajs_true=self.trajs_true, trajs_pred=trajs_pred_ns)

    def test_dimension_must_match_samples(self):
        with self.assertRaises(ValueError):
            EvalData(samples_true=np.zeros((10, 3)), trajs_true=self.trajs_true)


if __name__ == "__main__":
    unittest.main()
