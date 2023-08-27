# -*- coding: utf-8 -*-
from .data import load_gaussian, load_mnist
from .networks import Classifier, LinearClassifier
from .students import (
    ImitationStudent,
    OmniscientStudent,
    Student,
    SurrogateStudent,
)
from .teachers import (
    ImitationTeacher,
    OmniscientTeacher,
    SurrogateTeacher,
    Teacher,
)
from .train import (
    DatasetOptions,
    StudentOptions,
    TeacherOptions,
    TeachingType,
    train,
)
