import unittest
from Algorithm import Algorithm
import random
from scipy.optimize import linear_sum_assignment
import numpy as np
import json


class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.algos = Algorithm()

    def solve(self, matrix_ef):

        cost = np.array(matrix_ef)

        row_ind, col_ind = linear_sum_assignment(-1 * cost)

        return cost[row_ind, col_ind].sum()

    def test_Graph(self):
            for i in range(5, 100):
                matrix_ef = []
                for _ in range(i):
                    matrix_ef.append([0] * i)  

                for row in range(0, i):
                    for col in range(0, i):
                        matrix_ef[row][col] = int(random.randint(1, 10)) 

                print(matrix_ef, self.algos.graph(matrix_ef))  

                self.assertEqual(self.algos.graph(matrix_ef), self.solve(matrix_ef))
            

if __name__ == '__main__':
    unittest.main()
