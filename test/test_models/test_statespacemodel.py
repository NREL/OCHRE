import unittest
import datetime as dt
import numpy as np

from ochre.Models import StateSpaceModel

# inputs for SISO test
x0_1 = {"x1": 5}
u_defaults1 = {"u1": 0}
a1 = -2
b1 = 1

# inputs for MIMO test (3 states, 4 inputs, 2 outputs)
x0_2 = {f"x{i + 1}": i + 1 for i in range(3)}
u_defaults2 = {f"u{i + 1}": val for i, val in enumerate([5, 0, 0, 0])}
y2 = ["y1", "y2"]
np.random.seed(1)
a2 = np.random.randn(3, 3) / 10 - np.eye(3) / 2  # for PSD matrix
b2 = np.random.randn(3, 4)
c2 = np.random.randn(2, 3)

# Common simulation parameters required by Simulator base class
sim_params = {
    "start_time": dt.datetime(2020, 1, 1),
    "duration": dt.timedelta(hours=1),
    "verbosity": 0,  # suppress output
}


class SSModelTestCase(unittest.TestCase):
    """
    Test Case to test the StateSpaceModel class.
    """

    def setUp(self):
        self.model = StateSpaceModel(
            states=x0_1, inputs=u_defaults1, matrices=(a1, b1), time_res=dt.timedelta(seconds=2), **sim_params
        )

    def test_init(self):
        self.assertDictEqual(self.model.get_states(), x0_1)
        self.assertDictEqual(self.model.get_inputs(), u_defaults1)

        self.assertTupleEqual(self.model.A_c.shape, (1, 1))
        self.assertEqual(self.model.A_c[0, 0], -2)
        self.assertTupleEqual(self.model.B_c.shape, (1, 1))
        self.assertEqual(self.model.B_c[0, 0], 1)

    def test_to_discrete(self):
        # to_discrete now uses self.A_c and self.B_c internally, only takes optional time_res
        A, B = self.model.to_discrete()
        self.assertListEqual(A.tolist(), self.model.A.tolist())
        self.assertAlmostEqual(A[0, 0], 0.018, places=3)
        self.assertAlmostEqual(B[0, 0], 0.49, places=2)

        A, B = self.model.to_discrete(time_res=dt.timedelta(seconds=10))
        self.assertAlmostEqual(A[0, 0], 0, places=3)
        self.assertAlmostEqual(B[0, 0], 0.50, places=2)

    def test_update_inputs(self):
        # The new API uses control_signal in update_model to set inputs
        # update_inputs is for schedule-based inputs, not arbitrary dict inputs
        # Test that inputs are properly set via control_signal
        self.model.update_model(control_signal={"u1": 2})
        self.assertEqual(self.model.inputs[0], 2)

    def test_update(self):
        # test state change
        self.model.update()
        self.assertLess(self.model.states[0], x0_1["x1"])

        self.model.update(control_signal={"u1": 100})
        self.assertGreater(self.model.states[0], x0_1["x1"])

        # test steady state
        for _ in range(200):
            self.model.update()
        self.assertAlmostEqual(self.model.states[0], 0, places=3)

        for _ in range(200):
            self.model.update(control_signal={"u1": 2})
        self.assertAlmostEqual(self.model.states[0], -b1 / a1 * 2, places=3)


class LargeRCModelTestCase(unittest.TestCase):
    """
    Test Case to test the RCModel class.
    """

    def setUp(self):
        self.model = StateSpaceModel(
            states=x0_2,
            inputs=u_defaults2,
            outputs=y2,
            matrices=(a2, b2, c2),
            time_res=dt.timedelta(minutes=1),
            **sim_params,
        )

    def test_init(self):
        self.assertDictEqual(self.model.get_states(), x0_2)
        self.assertDictEqual(self.model.get_inputs(), u_defaults2)
        self.assertListEqual(self.model.output_names, y2)

        self.assertTupleEqual(self.model.A_c.shape, (3, 3))
        self.assertTupleEqual(self.model.B_c.shape, (3, 4))
        self.assertTupleEqual(self.model.C.shape, (2, 3))

    def test_reduce_model(self):
        # reduce_model now modifies the model in-place and returns None
        # test with reduced states
        self.model.reduce_model(reduced_states=2)
        self.assertEqual(self.model.nx, 2)
        self.assertTupleEqual(self.model.A_c.shape, (2, 2))
        self.assertTupleEqual(self.model.B_c.shape, (2, 4))
        self.assertTupleEqual(self.model.C.shape, (2, 2))

        # Reset model for next test
        self.setUp()

        # test with reduced_min_accuracy
        self.model.reduce_model(reduced_min_accuracy=0.2)
        self.assertEqual(self.model.nx, 1)
        self.assertTupleEqual(self.model.A_c.shape, (1, 1))
        self.assertTupleEqual(self.model.B_c.shape, (1, 4))
        self.assertTupleEqual(self.model.C.shape, (2, 1))

        # Save A for comparison
        A_first = self.model.A_c[0, 0]

        # Reset model for next test
        self.setUp()

        # test with input weights
        self.model.reduce_model(reduced_min_accuracy=0.2, input_weights=np.arange(4))
        self.assertEqual(self.model.nx, 1)
        self.assertNotEqual(A_first, self.model.A_c[0, 0])

    def test_update(self):
        # test state change
        self.model.update()
        self.assertLess(self.model.states[0], x0_2["x1"])
        self.assertLess(self.model.states[1], x0_2["x2"])

        # test output - just verify outputs exist and have been computed
        # (update_states parameter no longer exists)
        outputs = self.model.outputs
        self.assertNotEqual(outputs[0], 0)


if __name__ == "__main__":
    unittest.main()
