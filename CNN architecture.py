import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm




def my_EEGNet(nb_classes, Chans=32, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=1, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples,1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same', data_format='channels_last',
                    input_shape=(Chans, Samples,1),
                    use_bias=False)(input1)
    block1 = Conv2D(2*F1, (1,kernLength), padding='same', data_format='channels_last',
                    use_bias=False)(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('relu')(block1)
    block1 = MaxPooling2D((1, 4),data_format='channels_last')(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('relu')(block2)
    block2 = MaxPooling2D((1, 8),data_format='channels_last')(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)