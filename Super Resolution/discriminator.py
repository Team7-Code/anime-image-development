from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, PReLU, Add, UpSampling2D, LeakyReLU, Flatten, Dense, Activation


class discriminator(object):

    def __init__(self, HRDim, opt):
        self.HRDim = HRDim
        self.opt = opt

    def get_discriminator(self):
        input = Input(shape=(self.HRDim, self.HRDim, 3))

        conv = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(input)
        Act = LeakyReLU(alpha=0.2)
        conv = Act(conv)

        conv = Conv2D(64, kernel_size=(3, 3), strides=2, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(128, kernel_size=(3, 3), strides=2, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        conv = Conv2D(512, kernel_size=(3, 3), strides=2, padding='same')(conv)
        conv = BatchNormalization(momentum=0.5)(conv)
        Act = LeakyReLU(0.2)
        conv = Act(conv)

        flatten = Flatten()(conv)

        dense = Dense(1024)(flatten)
        Act = LeakyReLU(0.2)
        dense = Act(dense)

        output = Dense(1, activation='sigmoid')(dense)

        discriminator = Model(inputs=input, outputs=output)
        discriminator.compile(loss='binary_crossentropy', optimizer=self.opt)

        return discriminator
