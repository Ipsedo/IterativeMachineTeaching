import unittest

import torch as th

from iterative_machine_teaching.networks import LinearClassifier
from iterative_machine_teaching.student import OmniscientStudent


class TestStudent(unittest.TestCase):
    def setUp(self) -> None:
        self.__batch_size = th.randint(16, (1,))[0].item() + 1

    def test_example_difficulty(self):
        x = th.randn(self.__batch_size, 16)
        y = th.randint(2, (self.__batch_size,)).to(th.float)

        model = LinearClassifier(16)
        student = OmniscientStudent(model)

        out = student.example_difficulty(x, y)

        self.assertEqual(out.size()[0], self.__batch_size)
        self.assertEqual(out.size()[0], x.size()[0])
        self.assertEqual(out.size()[0], y.size()[0])

        self.assertEqual(len(out.size()), 1)

    def test_example_usefulness(self):
        x = th.randn(self.__batch_size, 16)
        y = th.randint(2, (self.__batch_size,)).to(th.float)

        model = LinearClassifier(16)
        student = OmniscientStudent(model)

        teacher = LinearClassifier(16)

        out = student.example_usefulness(teacher, x, y)

        self.assertEqual(out.size()[0], self.__batch_size)
        self.assertEqual(out.size()[0], x.size()[0])
        self.assertEqual(out.size()[0], y.size()[0])

        self.assertEqual(len(out.size()), 1)
