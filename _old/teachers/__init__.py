# -*- coding: utf-8 -*-
from .imitation_teacher import (
    ImitationDiffLinearTeacher,
    ImitationLinearTeacher,
    ImmitationConvTeacher,
)
from .omniscient_teacher import (
    OmniscientConvStudent,
    OmniscientConvTeacher,
    OmniscientLinearStudent,
    OmniscientLinearTeacher,
)
from .surrogate_teacher import (
    SurrogateConvStudent,
    SurrogateConvTeacher,
    SurrogateDiffLinearTeacher,
    SurrogateLinearStudent,
    SurrogateLinearTeacher,
)
from .utils import BaseConv, BaseLinear
