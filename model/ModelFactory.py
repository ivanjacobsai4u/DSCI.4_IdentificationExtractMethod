from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier

from model.CNNCodeDuplExtr import CNNCodeDuplExt


class ModelFactory:

    def make_model(self,model_type):
        return self._init_implementation(model_type)

    def _init_implementation(self,model_type):
        type_to_implementation = {
            'rf': RandomForestClassifier,
            'sgd': SGDClassifier,
            'gnb': GaussianNB,
            'lrc': LogisticRegression,
            'cnn': CNNCodeDuplExt,
            'mlp': MLPClassifier,
            'cnb': ComplementNB,
            'gbc': GradientBoostingClassifier,

        }
        implementation = type_to_implementation.get(model_type, None)
        return implementation()