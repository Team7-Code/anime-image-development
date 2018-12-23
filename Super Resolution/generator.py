from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, PReLU, Add, UpSampling2D


class generator(object):
    def __init__(self, LRDim, num_residual_blocks):
        self.LRDim = LRDim
        self.num_residual_blocks = num_residual_blocks

    def residual_blocks(self, layer):
        res = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(layer)
        res = BatchNormalization(momentum=0.5)(res)
        Act = PReLU(shared_axes=[1, 2])
        res = Act(res)
        res = Conv2D(64, kernel_size=(3, 3),strides=1, padding='same')(res)
        res = BatchNormalization(momentum=0.5)(res)
        res = Add()([res, layer])

        return res


    def get_generator(self):
        input = Input(shape=(self.LRDim, self.LRDim, 3))

        conv_1 = Conv2D(64, kernel_size=(9, 9), strides=1, padding='same')(input)
        Act = PReLU(shared_axes=[1, 2])
        conv_1 = Act(conv_1)

        layer = conv_1
        res=""
        for _ in range(self.num_residual_blocks):
            res = self.residual_blocks(layer)
            layer = res

        conv_2 = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(res)
        conv_2 = BatchNormalization(momentum=0.5)(conv_2)
        conv_2 = Add()([conv_2, conv_1])

        conv = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')(conv_2)
        conv = UpSampling2D(2)(conv)
        Act = PReLU(shared_axes=[1, 2])
        conv = Act(conv)


        conv = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')(conv)
        conv = UpSampling2D(2)(conv)
        Act = PReLU(shared_axes=[1, 2])
        conv = Act(conv)

        conv = Conv2D(3, kernel_size=(9, 9), padding='same', activation='tanh')(conv)

        generator = Model(inputs = input, outputs = conv)
        # generator.compile(loss=vgg_loss.VGG_loss, optimizer=opt)
        return generator
