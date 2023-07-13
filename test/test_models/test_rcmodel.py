import unittest
import datetime as dt
import math

from ochre.Models import RCModel
from ochre.Models.RCModel import transform_floating_node

# inputs for small RC test (1R1C test)
x0_1 = 5
u_defaults1 = [3, 0]
rc_params1 = {'R_INT_EXT': 2,
              'C_INT': 10}

# inputs for larger RC test (6R4C test)
x0_2 = [1, 2, 3, 4]
u_defaults2 = [5, 0, 0, 0, 0]
rc_params2 = {
    'R_1_E1': 1,
    'R_1_2': 1,
    'R_1_3': 1,
    'R_2_E2': 1,
    'R_3_4': 1,
    'R_4_E1': 1,
    'C_1': 200,
    'C_2': 200,
    'C_3': 500,
    'C_4': 200,
}


class RCModelTestCase(unittest.TestCase):
    """
    Test Case to test the RCModel class.
    """

    def setUp(self):
        self.model = RCModel(['EXT'], rc_params=rc_params1, time_res=dt.timedelta(seconds=2))

        # set initial x and u
        self.model.states[:] = [x0_1]
        self.model.inputs[:] = u_defaults1
        self.model.default_inputs[:] = u_defaults1

    def test_init(self):
        self.assertListEqual(self.model.state_names, ['T_INT'])
        self.assertListEqual(self.model.states.tolist(), [x0_1])
        self.assertListEqual(self.model.capacitances.tolist(), [rc_params1['C_INT']])
        self.assertListEqual(self.model.input_names, ['T_EXT', 'H_INT'])
        self.assertListEqual(self.model.inputs.tolist(), u_defaults1)

        self.assertTupleEqual(self.model.A.shape, (1, 1))
        self.assertAlmostEqual(self.model.A[0, 0], 0.905, places=3)
        # print(self.model.A)
        self.assertTupleEqual(self.model.B.shape, (1, 2))
        self.assertAlmostEqual(self.model.B[0, 0], 0.095, places=3)
        self.assertAlmostEqual(self.model.B[0, 1], 0.190, places=3)
        # print(self.model.B)

    def test_create_matrices(self):
        A, B = self.model.create_rc_matrices({'INT': 10}, {('INT', 'EXT'): 2}, ['INT'], ['EXT'])
        self.assertAlmostEqual(A[0, 0], -0.05)
        self.assertAlmostEqual(B[0, 0], 0.05)
        self.assertAlmostEqual(B[0, 1], 0.1)

        # test abstract
        self.model.create_rc_matrices({'INT': 10}, {('INT', 'EXT'): 2}, ['INT'], ['EXT'], return_abstract=True)

    def test_par(self):
        self.assertAlmostEqual(self.model.par(2, 2), 1)
        self.assertAlmostEqual(self.model.par(3, 3, 3), 1)
        self.assertAlmostEqual(self.model.par(5), 5)

    def test_solve_for_input(self):
        u_desired = self.model.solve_for_input('T_INT', 'H_INT', x0_1)
        self.assertAlmostEqual(u_desired, 1)

        # check with update
        self.model.update({'H_INT': u_desired})
        self.assertAlmostEqual(self.model.states[0], x0_1)


class LargeRCModelTestCase(unittest.TestCase):
    """
    Test Case to test the RCModel class.
    """

    def setUp(self):
        self.model = RCModel(['E1', 'E2'], rc_params=rc_params2, unused_inputs=['H_3'],
                             time_res=dt.timedelta(minutes=1))

        # set initial x and u
        self.model.states[:] = x0_2
        self.model.inputs[:] = u_defaults2
        self.model.default_inputs[:] = u_defaults2

    def test_init(self):
        self.assertListEqual(self.model.state_names, ['T_1', 'T_2', 'T_3', 'T_4'])
        self.assertListEqual(self.model.states.tolist(), x0_2)
        self.assertListEqual(self.model.capacitances.tolist(), [rc_params2['C_' + str(i + 1)] for i in range(4)])
        self.assertListEqual(self.model.input_names, ['T_E1', 'T_E2', 'H_1', 'H_2', 'H_4'])
        self.assertListEqual(self.model.inputs.tolist(), u_defaults2)

        self.assertTupleEqual(self.model.A.shape, (4, 4))
        self.assertAlmostEqual(self.model.A[0, 0], math.exp(-60*3 / 200), places=1)
        self.assertAlmostEqual(self.model.A[2, 2], math.exp(-60*2 / 500), places=1)
        self.assertAlmostEqual(self.model.A[0, 1], 0.1, places=1)
        self.assertAlmostEqual(self.model.A[0, 3], 0, places=1)
        self.assertAlmostEqual(self.model.A[0, :].sum() + self.model.B[0, :2].sum(), 1, places=2)
        # print(self.model.A)

        self.assertTupleEqual(self.model.B.shape, (4, 5))
        self.assertAlmostEqual(self.model.B[0, 0], 0.2, places=1)
        self.assertAlmostEqual(self.model.B[0, 2], 0.2, places=1)
        self.assertAlmostEqual(self.model.B[0, 3], 0.03, places=2)
        self.assertAlmostEqual(self.model.B[0, 4], 0, places=2)
        self.assertAlmostEqual(self.model.B[3, 4], 0.2, places=1)
        # print(self.model.B)

    def test_update(self):
        # test state change
        self.model.update({})
        self.assertGreater(self.model.states[0], x0_2[0])
        self.assertLess(self.model.states[1], x0_2[1])

        # test steady state
        for _ in range(200):
            self.model.update({})
        for x, y in zip(self.model.states, [3.6, 1.8, 4.1, 4.5]):
            self.assertAlmostEquals(x, y, places=1)

        for _ in range(200):
            self.model.update({'H_1': -3})
        for x, y in zip(self.model.states, [2, 1, 3, 4]):
            self.assertAlmostEquals(x, y, places=4)

    def test_time_res(self):
        fast_model = RCModel(['E1', 'E2'], rc_params=rc_params2, unused_inputs=['H_3'],
                             time_res=dt.timedelta(seconds=10))
        # set initial x and u
        fast_model.states[:] = x0_2
        fast_model.inputs[:] = u_defaults2
        fast_model.default_inputs[:] = u_defaults2

        # faster model should have larger A diagonal, smaller off diagonal elements and B elements
        self.assertGreater(fast_model.A[0, 0], self.model.A[0, 0])
        self.assertLess(fast_model.A[0, 1], self.model.A[0, 1])
        self.assertLess(fast_model.B[0, 0], self.model.B[0, 0])
        self.assertLess(fast_model.B[0, 2], self.model.B[0, 2])

        # test time constants
        for _ in range(2):
            self.model.update({})
        for _ in range(12):
            fast_model.update({})
        self.assertAlmostEqual(self.model.states[0], fast_model.states[0], places=3)
        self.assertAlmostEqual(self.model.states[2], fast_model.states[2], places=3)

    def test_solve_for_inputs(self):
        u_desired = self.model.solve_for_inputs(0, [2, 3], x0_2[0])
        self.assertAlmostEqual(u_desired, -11.44, places=2)

        self.model.update({'H_1': u_desired / 2,
                           'H_2': u_desired / 2})
        self.assertAlmostEqual(self.model.states[0], x0_2[0])

        # Test with u_ratios
        self.model.inputs = self.model.default_inputs.copy()
        u_desired = self.model.solve_for_inputs(0, [2, 3], x0_2[0], u_ratios=[0.75, 0.25])
        self.assertAlmostEqual(u_desired, -6.39, places=2)

        self.model.update({'H_1': u_desired * 3 / 4,
                           'H_2': u_desired * 1 / 4})
        self.assertAlmostEqual(self.model.states[0], x0_2[0])

    def test_solve_for_multi_inputs(self):
        self.model.setup_multi_input_solver(['T_1', 'T_2'], [{'H_1': 1}, {'H_2': 0.5, 'H_4': 0.5}])
        x, B, T = self.model.solver_params
        self.assertTupleEqual(B.shape, (2, 2))
        self.assertTupleEqual(T.shape, (5, 2))

        u_desired = self.model.solve_for_multi_inputs(x0_2[:2])
        self.assertEqual(len(u_desired), 5)
        self.assertEqual(u_desired[0], 0)
        self.assertAlmostEqual(u_desired[2], -6.95, places=2)
        self.assertEqual(u_desired[3], u_desired[4])

        self.model.inputs += u_desired
        self.model.update(reset_inputs=False)
        self.assertAlmostEqual(self.model.states[0], x0_2[0])
        self.assertAlmostEqual(self.model.states[1], x0_2[1])

    def test_transform_floating_node(self):
        resistors = {tuple(name.upper().split('_')[1:]): val for name, val in rc_params2.items() if name[0] == 'R'}
        self.assertEqual(len(resistors), 6)

        # 2R combination
        remove_3 = transform_floating_node('3', resistors)
        self.assertEqual(len(remove_3), 5)
        self.assertEqual(remove_3[('1', '2')], resistors[('1', '2')])
        self.assertAlmostEqual(remove_3[('1', '4')], 2)

        # 4R combination
        remove_1 = transform_floating_node('1', resistors)
        self.assertEqual(len(remove_1), 6)
        self.assertAlmostEqual(remove_1[('2', '3')], 3)
        self.assertNotIn(('1', '2'), remove_1)

        # check double combination is order-agnostic
        remove_31 = transform_floating_node('1', remove_3)
        remove_13 = transform_floating_node('3', remove_1)
        for node1, node2 in remove_31.keys():
            if (node1, node2) in remove_13:
                self.assertAlmostEqual(remove_13[(node1, node2)], remove_31[(node1, node2)])
            elif (node2, node1) in remove_13:
                self.assertAlmostEqual(remove_13[(node2, node1)], remove_31[(node1, node2)])
            else:
                self.fail('Missing key.')

        # test with 0 resistance
        resistors[('2', '3')] = 0
        result = transform_floating_node('3', resistors)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[('1', '2')], 1 / 2)
        self.assertAlmostEqual(result[('2', '4')], 1)
        self.assertAlmostEqual(result[('2', 'E2')], 1)


if __name__ == '__main__':
    unittest.main()
