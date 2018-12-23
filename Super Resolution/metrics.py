import numpy as np
from comtypes import named_property
from keras import backend as K

class metrics:
    def PSNR(self, y_true, y_pred):
        return -10*K.log(K.mean(K.square(y_pred-y_true)))/K.log(10.0)