from .teachers import Teacher, OmniscientTeacher, SurrogateTeacher, ImitationTeacher
from .students import Student, OmniscientStudent, SurrogateStudent, ImitationStudent

from .networks import Classifier, LinearClassifier

from .train import train, TeachingType

from .data import load_mnist, load_gaussian
