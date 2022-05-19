from .imitation_teacher import ImmitationConvTeacher, ImitationLinearTeacher, ImitationDiffLinearTeacher
from .omniscient_teacher import (
    OmniscientConvTeacher,
    OmniscientConvStudent,
    OmniscientLinearStudent,
    OmniscientLinearTeacher
)
from .surrogate_teacher import (
    SurrogateConvTeacher,
    SurrogateConvStudent,
    SurrogateLinearTeacher,
    SurrogateDiffLinearTeacher,
    SurrogateLinearStudent
)
from .utils import BaseLinear, BaseConv
