import unittest
import datetime as dt
import numpy as np

from ochre.Models import StateSpaceModel, ModelException

# inputs for SISO test
x0_1 = {'x1': 5}
u_defaults1 = {'u1': 0}
a1 = -2
b1 = 1

# inputs for MIMO test (3 states, 4 inputs, 2 outputs)
x0_2 = {f'x{i + 1}': i + 1 for i in range(3)}
u_defaults2 = {f'u{i + 1}': val for i, val in enumerate([5, 0, 0, 0])}
y2 = ['y1', 'y2']
np.random.seed(1)
a2 = np.random.randn(3, 3) / 10 - np.eye(3) / 2  # for PSD matrix
b2 = np.random.randn(3, 4)
c2 = np.random.randn(2, 3)


class SSModelTestCase(unittest.TestCase):
    """
    Test Case to test the StateSpaceModel class.
    """

    def setUp(self):
        self.model = StateSpaceModel(states=x0_1, inputs=u_defaults1, matrices=(a1, b1),
                                     time_res=dt.timedelta(seconds=2))

    def test_init(self):
        self.assertDictEqual(self.model.get_states(), x0_1)
        self.assertDictEqual(self.model.get_inputs(), u_defaults1)

        self.assertTupleEqual(self.model.A_c.shape, (1, 1))
        self.assertEqual(self.model.A_c[0, 0], -2)
        self.assertTupleEqual(self.model.B_c.shape, (1, 1))
        self.assertEqual(self.model.B_c[0, 0], 1)

    def test_to_discrete(self):
        A, B = self.model.to_discrete(self.model.A_c, self.model.B_c, self.model.time_res)
        self.assertListEqual(A.tolist(), self.model.A.tolist())
        self.assertAlmostEqual(A[0, 0], 0.018, places=3)
        self.assertAlmostEqual(B[0, 0], 0.49, places=2)

        A, B = self.model.to_discrete(self.model.A_c, self.model.B_c, dt.timedelta(seconds=10))
        self.assertAlmostEqual(A[0, 0], 0, places=3)
        self.assertAlmostEqual(B[0, 0], 0.50, places=2)

    def test_update_inputs(self):
        # test with bad input
        with self.assertRaises(ModelException):
            self.model.update_inputs({'bad_input': 1, 'bad_input2': 2})

        # test with good input
        self.model.update_inputs({'u1': 2})
        self.assertDictEqual(self.model.get_inputs(), {'u1': 2})

        # test with list
        self.model.update_inputs([5])
        self.assertDictEqual(self.model.get_inputs(), {'u1': 5})

        # test no reset
        self.model.update_inputs({})
        self.assertDictEqual(self.model.get_inputs(), {'u1': 5})

    def test_update(self):
        # test state change
        self.model.update()
        self.assertLess(self.model.states[0], x0_1['x1'])

        self.model.update({'u1': 100})
        self.assertGreater(self.model.states[0], x0_1['x1'])

        # test without resetting inputs
        self.model.update(reset_inputs=False)
        self.assertEqual(self.model.get_inputs(), {'u1': 100})

        # test without updating states
        states = self.model.states.copy()
        out = self.model.update(update_states=False)
        self.assertIsNotNone(out)
        self.assertListEqual(self.model.states.tolist(), states.tolist())

        # test steady state
        for _ in range(200):
            self.model.update()
        self.assertAlmostEqual(self.model.states[0], 0, places=3)

        for _ in range(200):
            self.model.update({'u1': 2})
        self.assertAlmostEqual(self.model.states[0], -b1 / a1 * 2, places=3)


class LargeRCModelTestCase(unittest.TestCase):
    """
    Test Case to test the RCModel class.
    """

    def setUp(self):
        self.model = StateSpaceModel(states=x0_2, inputs=u_defaults2, outputs=y2, matrices=(a2, b2, c2),
                                     time_res=dt.timedelta(minutes=1))

    def test_init(self):
        self.assertDictEqual(self.model.get_states(), x0_2)
        self.assertDictEqual(self.model.get_inputs(), u_defaults2)
        self.assertListEqual(self.model.output_names, y2)

        self.assertTupleEqual(self.model.A_c.shape, (3, 3))
        self.assertTupleEqual(self.model.B_c.shape, (3, 4))
        self.assertTupleEqual(self.model.C.shape, (2, 3))

    def test_reduce_model(self):
        # test with reduced states
        x, A, B, C = self.model.reduce_model(reduced_states=2)
        self.assertEqual(len(x), 2)
        self.assertTupleEqual(A.shape, (2, 2))
        self.assertTupleEqual(B.shape, (2, 4))
        self.assertTupleEqual(C.shape, (2, 2))

        # test with reduced_min_accuracy
        x, A, B, C = self.model.reduce_model(reduced_min_accuracy=0.2)
        self.assertEqual(len(x), 1)
        self.assertTupleEqual(A.shape, (1, 1))
        self.assertTupleEqual(B.shape, (1, 4))
        self.assertTupleEqual(C.shape, (2, 1))

        # test with input weights
        x2, A2, B2, C2 = self.model.reduce_model(reduced_min_accuracy=0.2, input_weights=np.arange(4))
        self.assertEqual(len(x2), 1)
        self.assertNotEqual(A[0, 0], A2[0, 0])

    def test_update(self):
        # test state change
        self.model.update()
        self.assertLess(self.model.states[0], x0_2['x1'])
        self.assertLess(self.model.states[1], x0_2['x2'])

        # test output comparison
        outputs = self.model.update(update_states=False)
        self.assertNotEqual(outputs[0], 0)

        self.model.update()
        self.assertListEqual(self.model.outputs.tolist(), outputs.tolist())


if __name__ == '__main__':
    unittest.main()
