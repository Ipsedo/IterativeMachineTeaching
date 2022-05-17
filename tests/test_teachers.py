import unittest

import torch as th

from iterative_machine_teaching.networks import LinearClf
from iterative_machine_teaching.teachers import OmniscientTeacher
from iterative_machine_teaching.student import OmniscientStudent


class TestTeacher(unittest.TestCase):
    def setUp(self) -> None:
        self.__top_n = th.randint(16, (1,))[0].item() + 1
        self.__batch_size = th.randint(16, (1,))[0].item() + self.__top_n
        self.__nb_example = self.__batch_size + th.randint(256, (1,))[0].item()

    def test_select_n_example(self):
        x = th.randn(self.__nb_example, 16)
        y = th.randint(2, (self.__nb_example,)).to(th.float)

        s_model = LinearClf(16)
        student = OmniscientStudent(s_model)

        t_model = LinearClf(16)
        teacher = OmniscientTeacher(t_model, self.__batch_size)

        top_data, top_label = teacher.select_n_examples(student, x, y, self.__top_n)

        self.assertEqual(top_data.size()[0], self.__top_n)
        self.assertEqual(top_label.size()[0], self.__top_n)

        self.assertEqual(top_data.size()[1], x.size()[1])
