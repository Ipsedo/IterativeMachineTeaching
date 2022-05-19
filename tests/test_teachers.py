import unittest
from typing import Tuple

import torch as th

from iterative_machine_teaching.networks import LinearClassifier
from iterative_machine_teaching.teachers import Teacher, OmniscientTeacher, SurrogateTeacher, ImitationTeacher
from iterative_machine_teaching.students import Student, OmniscientStudent, SurrogateStudent, ImitationStudent


class TestTeacher(unittest.TestCase):
    def setUp(self) -> None:
        self.__top_n = th.randint(16, (1,))[0].item() + 1
        self.__batch_size = th.randint(16, (1,))[0].item() + self.__top_n
        self.__nb_example = self.__batch_size + th.randint(256, (1,))[0].item()

    def __compute_select_n_example(
            self,
            teacher: Teacher,
            student: Student,
            x: th.Tensor,
            y: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        return teacher.select_n_examples(student, x, y, self.__top_n)

    def __check_dim(self, top_data: th.Tensor, top_label: th.Tensor, x: th.Tensor):
        self.assertEqual(top_data.size()[0], self.__top_n)
        self.assertEqual(top_label.size()[0], self.__top_n)

        self.assertEqual(top_data.size()[1], x.size()[1])

    def test_omniscient_select_n_example(self):
        x = th.randn(self.__nb_example, 16)
        y = th.randint(3, (self.__nb_example,))

        s_model = LinearClassifier(16, 3)
        student = OmniscientStudent(s_model, 1e-3)

        t_model = LinearClassifier(16, 3)
        teacher = OmniscientTeacher(t_model, 1e-3, self.__batch_size)

        top_data, top_label = self.__compute_select_n_example(teacher, student, x, y)

        self.__check_dim(top_data, top_label, x)

    def test_surrogate_select_n_example(self):
        x = th.randn(self.__nb_example, 16)
        y = th.randint(3, (self.__nb_example,))

        s_model = LinearClassifier(16, 3)
        student = SurrogateStudent(s_model, 1e-3)

        t_model = LinearClassifier(16, 3)
        teacher = SurrogateTeacher(t_model, 1e-3, self.__batch_size)

        top_data, top_label = self.__compute_select_n_example(teacher, student, x, y)

        self.__check_dim(top_data, top_label, x)

    def test_imitation_select_n_example(self):
        x = th.randn(self.__nb_example, 16)
        y = th.randint(3, (self.__nb_example,))

        s_model = LinearClassifier(16, 3)
        student = ImitationStudent(s_model, 1e-3)

        t_model = LinearClassifier(16, 3)
        teacher = ImitationTeacher(t_model, 1e-3, self.__batch_size)

        top_data, top_label = self.__compute_select_n_example(teacher, student, x, y)

        self.__check_dim(top_data, top_label, x)
