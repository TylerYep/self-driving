import sys
from .naive import NaiveConditionedCNN
from .resnet import PretrainedResNet

def Model(model_name):
    ''' Retrieves class initializer from its string name. '''
    return getattr(sys.modules[__name__], model_name)()