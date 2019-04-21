import numpy as np
from keras.datasets import mnist
import random
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 8
plt.rcParams['figure.figsize'] = (8,8)

from sklearn import preprocessing
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D, MaxPooling2D,LeakyReLU,Conv2DTranspose
from keras.layers import Input, Lambda, concatenate, Add, Concatenate
from keras import backend as K

from keras.optimizers import Adam, RMSprop

from privacy_net.utils.module import u_tail
from privacy_net.utils.module import y_tail
from privacy_net.utils.module import InstanceNormalization2D
from privacy_net.utils.module import residual_block
from privacy_net.models.privacy_net_model import privacyNet


def generator(img):
    #     def tileAttr(x):
    #         x = tf.expand_dims(x, axis = 1)
    #         x = tf.expand_dims(x, axis = 2)
    #         return tf.tile(x, [1, size, size, 1])

    #     y = Concatenate()([img, Lambda(tileAttr)(attr)])

    y = Conv2D(32, 5, padding="same", kernel_initializer='he_normal')(img)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)

    y = Conv2D(64, 3, strides=2, padding="same", kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)

    y = Conv2D(128, 3, strides=2, padding="same", kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)

    y = residual_block(y, 128, 3)
    y = residual_block(y, 128, 3)
    y = residual_block(y, 128, 3)
    y = residual_block(y, 128, 3)

    y = Conv2DTranspose(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)

    y = Conv2DTranspose(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)

    y = Conv2D(1, 5, strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = Activation('tanh')(y)
    return y

def discriminator(img, u_size, y_size, u_feat_dims=None, y_feat_dims=None, activation='tanh'):
    #     def tileAttr2(x):
    #         x = tf.expand_dims(x, axis = 1)
    #         x = tf.expand_dims(x, axis = 2)
    #         return tf.tile(x, [1, size//64, size//64, 1])

    y = Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(img)
    y = LeakyReLU(alpha=0.2)(y)  # 14 14 32
    y = Dropout(0.25)(y)

    y = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = ZeroPadding2D(padding=((0, 1), (0, 1)))(y)
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)  # 8 8 64
    y = Dropout(0.25)(y)

    y = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)  # 4 4 128
    y = Dropout(0.25)(y)

    y = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)  # 4 4 256
    y = Dropout(0.25)(y)

    y = Conv2D(512, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)  # 2 2 512
    y = Dropout(0.25)(y)

    pred_real = Conv2D(1, kernel_size=2, strides=1, kernel_initializer='he_normal')(y)
    pred_f_u = []
    pred_g_y = []
    if not u_feat_dims:
        u_feat_dims = [2 for _ in range(u_size)]
    if not y_feat_dims:
        y_feat_dims = [2 for _ in range(y_size)]

    for i in range(u_size):
        temp_u = Conv2D(u_feat_dims[i], kernel_size=2, strides=1, kernel_initializer='he_normal',
                        activation=activation)(y)
        temp_u = Flatten()(temp_u)
        pred_f_u.append(temp_u)

    for i in range(y_size):
        temp_y = Conv2D(y_feat_dims[i], kernel_size=2, strides=1, kernel_initializer='he_normal',
                        activation=activation)(y)
        temp_y = Flatten()(temp_y)
        pred_g_y.append(temp_y)

    return pred_real, pred_f_u, pred_g_y

def save_some(im_list,name):
    im_list = np.squeeze(im_list)
    num_list = len(im_list)
    plt.figure(figsize=(3 * num_list, 4))
    for i, array in enumerate(im_list):
        plt.subplot(1, len(im_list), i + 1)
        plt.imshow(((array + 1) * 127.5).astype(np.uint8), cmap='gray')
        plt.axis('off')
    plt.savefig(name)


class reversedMNISTGenerator():
    def __init__(self, batch_size=16):

        self.input_shape = (28, 28, 1)
        self.batch_size = batch_size
        (self.x_train, self.digit_train), (self.x_test, self.digit_test) = mnist.load_data()
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)

        self.x_train = np.concatenate((self.x_train, 255 - self.x_train), axis=0)
        self.x_test = np.concatenate((self.x_test, 255 - self.x_test), axis=0)
        self.digit_train = np.concatenate((self.digit_train, self.digit_train), axis=0)
        self.digit_test = np.concatenate((self.digit_test, self.digit_test), axis=0)

        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

        self.u_train, self.u_test = self.load_public_label()
        self.y_train, self.y_test = self.load_sensitive_label()

        self.steps_per_epoch = self.default_training_steps
        self.test_steps = self.default_testing_steps

        self.cur_train_step = 0
        self.cur_test_step = 0
        self.train_ind_list = np.array([i for i in range(self.train_size)])
        self.test_ind_list = np.array([i for i in range(self.test_size)])
        random.seed(1234)
        random.shuffle(self.train_ind_list)
        random.shuffle(self.test_ind_list)

    # property is a getter
    @property
    def default_training_steps(self):
        # make sure train with full batch size which means each epoch less than batch size images being used.
        return int(np.floor(self.train_size / self.batch_size))

    @property
    def default_testing_steps(self):
        # make sure train with full batch size which means each epoch less than batch size images being used.
        return int(np.floor(self.test_size / self.batch_size))

    def load_public_label(self):
        # prepare utility labels
        #         u_train = np.zeros_like(self.digit_train)
        #         u_test = np.zeros_like(self.digit_test)

        #         u_train[self.digit_train % 2 == 1] = 1
        #         u_test[self.digit_test % 2 == 1] = 1

        return self.digit_train, self.digit_test

    def load_sensitive_label(self):
        # prepare privacy labels
        y_train = np.zeros_like(self.digit_train)
        y_test = np.zeros_like(self.digit_test)

        y_train[int(self.train_size / 2):] = 1
        y_test[int(self.test_size / 2):] = 1
        return y_train, y_test

    def nextTrain(self):
        while True:

            # set self.cur_train_ind
            # cur_train_ind has indices of batch_sizes, this indices correpond to the image ids
            self.fetchCurrentTrainInd()
            train_x = self.x_train[self.cur_train_ind]
            train_x = train_x / 127.5 - 1

            train_u = np.zeros((self.batch_size, 10))
            for i in range(self.batch_size):
                train_u[i][self.u_train[self.cur_train_ind[i]]] = 1
            train_y = np.zeros((self.batch_size, 2))
            for i in range(self.batch_size):
                train_y[i][self.y_train[self.cur_train_ind[i]]] = 1

            inputs = {'input_x': train_x, 'input_y': [train_y], 'input_u': [train_u]}
            output = np.zeros((self.batch_size, 1))
            self.cur_train_step += 1
            yield inputs, output

    def fetchCurrentTrainInd(self):
        # this function find cur_train_ind for each step
        # it read batch_size's image_ids, starting from cur_train_step(batch_size,augment_ratio)

        # if self.cur_train_step%self.augment_ratio != 0:
        #     return

        # when epoch end it will shuffle image_ids and reset cur_train_step, namely, read from the start
        if self.cur_train_step >= self.steps_per_epoch:
            random.shuffle(self.train_ind_list)
            # print("shuffling")
            self.cur_train_step = 0

        cur_train_start = int(self.cur_train_step) * self.batch_size
        cur_train_end = cur_train_start + self.batch_size
        if cur_train_end > self.train_size:
            cur_train_end = self.train_size
        self.cur_train_ind = self.train_ind_list[np.arange(cur_train_start, cur_train_end).astype(int)]

    def nextVal(self):
        while True:
            if self.cur_test_step == self.test_steps:
                self.cur_test_step = 0
                random.shuffle(self.test_ind_list)
            cur_test_start = int(self.cur_test_step) * self.batch_size
            cur_test_end = cur_test_start + self.batch_size
            if cur_test_end > self.test_size:
                cur_test_end = self.test_size
            self.cur_test_ind = self.test_ind_list[np.arange(cur_test_start, cur_test_end)]

            test_x = self.x_test[self.cur_test_ind]
            test_x = test_x / 127.5 - 1

            test_u = np.zeros((self.batch_size, 10))
            for i in range(self.batch_size):
                test_u[i][self.u_test[self.cur_test_ind[i]]] = 1
            test_y = np.zeros((self.batch_size, 2))
            for i in range(self.batch_size):
                test_y[i][self.y_test[self.cur_test_ind[i]]] = 1

            self.cur_test_step += 1
            inputs = {'input_x': test_x, 'input_y': [test_y], 'input_u': [test_u]}
            output = np.zeros((self.batch_size, 1))
            yield inputs, output


class privacyNetV3(privacyNet):

    def __init__(self, u_input_dims=[2, 2, 2, 2], y_input_dims=[2], u_feat_dims=[2, 2, 2, 2],
                 y_feat_dims=[2], mode=0,
                 **kwargs):
        self.u_input_dims = u_input_dims
        self.y_input_dims = y_input_dims
        self.u_feat_dims = u_feat_dims
        self.y_feat_dims = y_feat_dims
        self.mode = mode  # 0->generatorV2 1->generatorV3 2->generatorV4
        super(privacyNetV3, self).__init__(**kwargs)

    def build_model(self):

        """Build Discriminator and Generator and Transformer, seperately.
        This method requires no inputs and returns nothing.
        self.d_model, self.g_model, self.transform_tail_models are goals for this method.
        NOTE: The following 'Output's and 'Input's are used for introducing model specs.

        # D model arguments:
            - d_model: Discriminator part.
        Input:
            - attr_y_input: list of Input Tensors. sensitive labels, Y.
            - attr_u_input(Optional): list, of Input Tensors. public labels, U.
            - img_input: Input Tensor, shared with g_model. raw images, X.
        Output:
            - loss_y_output: list, H-score between imgs and sensitive labels. H_score.
            - loss_u_output: list, H-score between imgs and public labels. Might be None. H_score
            - pred_real: predictions of real images. prob.
            - pred_f_y: list, feature of Y. y(X).
            - pred_f_u: list, feature of U. u(X).

        # G model:
            - g_model: Generator part.
        Input:
            - img_input: Input Tensor, shared with d_model. raw images, X.
        Output:
            - img_output: transformed images. transformed images, Z.

        # T models:
            - transform_tail_models: list of T models.
        Input:
            - transform_y_input: list of input of third networks in order to filter
                                 sensitive information. y(Z)
        Output:
            - transform_y_tail: list. h(y(Z))

        """
        ################## Part I ######################
        ## start building D model
        self.img_input = Input(self.image_size)
        self.attr_y_input = []  # placeholder for every sensitive labels
        self.attr_u_input = []  # placeholder for every public labels

        for i in range(self.y_size):
            self.attr_y_input.append(Input((self.y_input_dims[i],)))
        for j in range(self.u_size):
            self.attr_u_input.append(Input((self.u_input_dims[j],)))

        self.pred_real, self.pred_f_u, self.pred_f_y = discriminator(self.img_input,
                                                                     self.u_size,
                                                                     self.y_size,
                                                                     self.u_feat_dims,
                                                                     self.y_feat_dims,
                                                                     activation=self.activation)

        # All tail networks are separated
        self.attr_y_tail = []  # holding tails network for sensitive_attrs
        self.attr_u_tail = []  # holding tails network for public_attrs

        for i in range(self.y_size):
            self.attr_y_tail.append(y_tail(self.attr_y_input[i],
                                           self.y_feat_dims[i],
                                           activation=self.activation))

        for j in range(self.u_size):
            self.attr_u_tail.append(u_tail(self.attr_u_input[j],
                                           self.u_feat_dims[j],
                                           activation=self.activation))

        self.loss_y_output = []  # containing sensitive H-score
        self.loss_u_output = []  # containing public H-score

        for i in range(self.y_size):
            self.loss_y_output.append(Lambda(self.neg_hscore)([self.pred_f_y[i], self.attr_y_tail[i]]))

        for j in range(self.u_size):
            self.loss_u_output.append(Lambda(self.neg_hscore)([self.pred_f_u[j], self.attr_u_tail[j]]))

        self.d_model = Model(inputs=[self.img_input] + self.attr_u_input + self.attr_y_input,
                             outputs=[self.pred_real] + self.loss_u_output + self.loss_y_output
                                     + self.pred_f_u + self.pred_f_y + self.attr_u_tail + self.attr_y_tail)
        if self.d_model_dir:
            self.d_model.load_weights(self.d_model_dir)  # load pretrained model if avialable
        ## end building D model

        ################## Part II ######################
        ## start building G model
        if self.mode == 0:
            self.img_output = generatorV2(self.img_input)  # pass image into generator
        elif self.mode == 1:
            self.img_output = generatorV3(self.img_input)  # pass image into generator
        elif self.mode == 2:
            self.img_output = generatorV4(self.img_input)  # pass image into generator
        else:
            self.img_output = generator(self.img_input)  # pass image into generator

        self.g_model = Model(inputs=[self.img_input], outputs=[self.img_output])

        if self.g_model_dir:
            self.g_model.load_weights(self.g_model_dir)  # load pretrained model if avialable
        ## end building G model

        ################## Part III ######################
        ## start building T model
        self.transform_y_input = []  # T model inputs
        for i in range(self.y_size):
            self.transform_y_input.append(Input((self.y_feat_dims[i],)))

        self.transform_y_tail = []  # T model outputs
        for i in range(self.y_size):
            self.transform_y_tail.append(y_tail(self.transform_y_input[i],
                                                self.y_feat_dims[i],
                                                activation=self.activation))
        self.transform_tail_models = []
        for i in range(self.y_size):
            self.transform_tail_models.append(Model(inputs=[self.transform_y_input[i]],
                                                    outputs=[self.transform_y_tail[i]]))
        if self.t_model_y_dir:
            for ind_y in range(self.y_size):
                self.transform_tail_models[ind_y].load_weights(self.t_model_y_dir[ind_y])
        ## end building T model

    def build_loss(self):
        """Build Discriminator and Generator and Transformer loss, itegrating G,D,T into privacy_net model.
        This project uses three steps training, D->T->G. Thus, three losses are introduced accordingly.

        # Network Arguments:

            - img_a: Input Tensor. Raw images.
            - attr_y: list, Input Tensors. Sensitive labels.
            - attr_u: list, Input Tensors. Public labels.
            - img_a_trans: Output Tensor of G. Transformed images.
            - pred_real(local): List, Output tensor of D. On raw images.
            Details: pred_real
            |0       |1:1+u_size   |1+u_size:1+u_size+y_size|1+u_size+y_size:1+u_size*2+y_size|1+u_size*2+y_size:1+u_size*2+y_size*2|
            1+u_size*2+y_size*2:1+u_size*3+y_size*2|1+u_size*3+y_size*2:1+u_size*3+y_size*3|
            |pred_dis|loss_u_hscore|      loss_y_hscore     |           u(X)                  |               y(X)                  |
                               f(U)                |                   g(Y)                |
            - pred_trans(local): List, Output tensor of D. On transformed images.
            Details: pred_trans
            |0       |1:1+u_size   |1+u_size:1+u_size+y_size|1+u_size+y_size:1+u_size*2+y_size|1+u_size*2+y_size:1+u_size*2+y_size*2|
            1+u_size*2+y_size*2:1+u_size*3+y_size*2|1+u_size*3+y_size*2:1+u_size*3+y_size*3|
            |pred_dis|loss_u_hscore|      loss_y_hscore     |           u(Z)                  |               y(Z)                  |
                               f(U)                |                   g(Y)                |
            - h_z(local): list, Output Tensors of T. h(y(Z)).

        D loss Arguments:

        G loss Arguments:

        T loss Arguments:

        Print Arguments:

        """

        # Same implementation with neg_hscore, in case of mixing up with Lambda layer.
        def neg_hscore_loss(x):
            """
            neg hscore loss implementation
            """
            f = x[0]
            g = x[1]
            f0 = f - K.mean(f, axis=0)
            g0 = g - K.mean(g, axis=0)
            corr = tf.reduce_mean(tf.reduce_sum(tf.multiply(f0, g0), 1))
            cov_f = K.dot(K.transpose(f0), f0) / K.cast(K.shape(f0)[0] - 1, dtype='float32')
            cov_g = K.dot(K.transpose(g0), g0) / K.cast(K.shape(g0)[0] - 1, dtype='float32')
            return - corr + tf.trace(K.dot(cov_f, cov_g)) / 2

        ########### Part 0 ################
        ## Connecting three models
        self.img_a = Input(shape=self.image_size)  # Raw images tensor.

        self.attr_y = []  # sensitive_label tensor.
        self.attr_u = []  # public_label tensor.
        for i in range(self.y_size):
            self.attr_y.append(Input((self.y_input_dims[i],)))
        for j in range(self.u_size):
            self.attr_u.append(Input((self.u_input_dims[j],)))

        self.img_a_trans = self.g_model(self.img_a)  # Transformed image Tensor.
        pred_real = self.d_model([self.img_a] + self.attr_u + self.attr_y)  # D output tensor.
        pred_trans = self.d_model([self.img_a_trans] + self.attr_u + self.attr_y)  # D output tensor.

        h_z = []  # h(y(Z)) tensor.
        for i in range(self.y_size):
            h_z.append(self.transform_tail_models[i](pred_trans[1 + self.u_size * 2 + self.y_size + i]))
        ## end Connecting three models

        ############## Part I #################
        ## build D loss
        # discriminate real or fake, follows WGAN.
        d_loss_real = K.mean(K.square(K.ones_like(pred_real[0]) - pred_real[0]), axis=-1)
        d_loss_fake = K.mean(K.square(K.zeros_like(pred_trans[0]) - pred_trans[0]), axis=-1)

        # gradient panelty: WGAN-gp
        self.eps_input = K.placeholder(shape=(None, 1, 1, 1))
        mixed_input = Input(shape=self.image_size,
                            tensor=self.eps_input * self.img_a + (1 - self.eps_input) * self.img_a_trans)
        gp_pred = self.d_model([mixed_input] + self.attr_u + self.attr_y)
        grad_mixed = K.gradients(gp_pred[0], [mixed_input])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
        d_loss_gp = K.mean(K.square(norm_grad_mixed - 1))

        # compute u_loss in total
        for i in range(1, 1 + self.u_size):
            if i == 1:
                d_loss_u_total = pred_real[i]
            else:
                d_loss_u_total += pred_real[i]

        # compute y_loss in total
        for j in range(1 + self.u_size, 1 + self.u_size + self.y_size):
            if j == 1 + self.u_size:
                d_loss_y_total = pred_real[j]
            else:
                d_loss_y_total += pred_real[j]

        # d_loss = u_loss+y_loss+loss_real+loss_fake+loss_gp
        if self.u_size == 0:
            self.d_loss = d_loss_real + d_loss_fake + self.lambda_cls * ((
                                                                                 1. / self.y_size) * d_loss_y_total) + self.lambda_gp * d_loss_gp
        else:
            self.d_loss = d_loss_real + d_loss_fake + self.lambda_cls * ((1 / self.u_size) * d_loss_u_total + (
                    1. / self.y_size) * d_loss_y_total) + self.lambda_gp * d_loss_gp
        ## end build D loss

        ############## Part II #################
        ## build G
        # loss real
        g_loss_fake = K.mean(K.square(K.ones_like(pred_trans[0]) - pred_trans[0]), axis=-1)

        # loss_u is Hscore(u(X),u(Z))
        for i_g in range(1 + self.u_size + self.y_size, 1 + self.u_size * 2 + self.y_size):
            if i_g == 1 + self.u_size + self.y_size:
                g_loss_u_total = neg_hscore_loss([pred_trans[i_g], pred_real[i_g]])
            else:
                g_loss_u_total += neg_hscore_loss([pred_trans[i_g], pred_real[i_g]])

        # loss_u is Hscore(y(X),h(y(Z)))
        for j_g in range(1 + self.u_size * 2 + self.y_size, 1 + self.u_size * 2 + self.y_size * 2):
            if j_g == 1 + self.u_size * 2 + self.y_size:
                g_loss_y_total = neg_hscore_loss([h_z[j_g - (1 + self.u_size * 2 + self.y_size)], pred_real[j_g]])
            else:
                g_loss_y_total += neg_hscore_loss([h_z[j_g - (1 + self.u_size * 2 + self.y_size)], pred_real[j_g]])

        if self.u_size == 0:
            self.g_loss = g_loss_fake - self.lambda_cls * ((
                                                                   1. / self.y_size) * self.gamma * g_loss_y_total)
        else:
            self.g_loss = g_loss_fake + self.lambda_cls * ((1. / self.u_size) * g_loss_u_total - (
                    1. / self.y_size) * self.gamma * g_loss_y_total)
        ## end build G loss

        ############## Part III ###############
        ## build T loss
        self.transform_loss = []  # list, transform loss.

        # H-score of Hscore[h(y(Z)),y(X)]
        for i in range(self.y_size):
            self.transform_loss.append(neg_hscore_loss([h_z[i], pred_real[1 + self.u_size * 2 + self.y_size + i]]))
        ## end build T loss

        # save list for detail printing.
        self.d_loss_list_print = []
        self.d_loss_list_print.append(d_loss_real)
        self.d_loss_list_print.append(d_loss_fake)
        if self.u_size > 0:
            self.d_loss_list_print.append(d_loss_u_total)
        self.d_loss_list_print.append(d_loss_y_total)
        self.d_loss_list_print.append(d_loss_gp)

        self.g_loss_list_print = []
        self.g_loss_list_print.append(g_loss_fake)
        if self.u_size > 0:
            self.g_loss_list_print.append(g_loss_u_total)
        self.g_loss_list_print.append(g_loss_y_total)

    def build_optimizer(self):

        self.g_training_updates = Adam(lr=self.lr, decay=5e-9, beta_1=self.b1, beta_2=self.b2).get_updates(
            self.g_model.trainable_weights, [], self.g_loss)
        self.g_train = K.function([self.img_a] + self.attr_u + self.attr_y, self.g_loss_list_print,
                                  self.g_training_updates)

        self.transform_train = []
        for i in range(self.y_size):
            self.transform_training_updates = Adam(lr=self.lr, decay=5e-9, beta_1=self.b1, beta_2=self.b2).get_updates(
                self.transform_tail_models[i].trainable_weights, [], self.transform_loss[i])
            self.transform_train.append(K.function([self.img_a] + self.attr_u + self.attr_y,
                                                   [self.transform_loss[i]], self.transform_training_updates))

        self.d_training_updates = Adam(lr=self.lr, decay=5e-9, beta_1=self.b1, beta_2=self.b2).get_updates(
            self.d_model.trainable_weights, [], self.d_loss)
        self.d_train = K.function([self.img_a, self.eps_input] + self.attr_u + self.attr_y, self.d_loss_list_print,
                                  self.d_training_updates)

    def hscore_accu(self, pred, pred_dummy, pu, py, u_labels, y_labels):
        """
        For testing images only. This function compute the accuracy of each private/public attributes.
        Note: At early stage, the evaluation model is still under training. One can simply look at the accuracy curve
        of each attribute(on original images) to determine whether the models are ready.
        One convenience is that we do not need to train auxiliary models to evaluate.
        Input:
            - pred: output of d_model
            - pred_dummy: to extract output of a-models(np.eye(2) as input).
            Details:
            |0       |1:1+u_size   |1+u_size:1+u_size+y_size|1+u_size+y_size:1+u_size*2+y_size|1+u_size*2+y_size:1+u_size*2+y_size*2|
            1+u_size*2+y_size*2:1+u_size*3+y_size*2|1+u_size*3+y_size*2:1+u_size*3+y_size*3|
            |pred_dis|loss_u_hscore|      loss_y_hscore     |           u(X)                  |               y(X)                  |
                               f(U)                |                   g(Y)                |
            - py: list, labels marginal distribution. (training data)
            - pu: list, labels marginal distribution. (training data)
            - y_labels: list, labels of private attrs
            - u_labels: list, models of public attrs
        Output:
            - d_score: 1 - Loss_real
            - u_accu: list, utility accuracy on each attrs. JUST LIKE LOG-LOSS!!!
            - p_accu: list, privacy accuracy on each attrs.
        """
        p_accu = []
        for ind_y in range(self.y_size):
            y_X = pred[1 + self.u_size * 2 + self.y_size + ind_y]  # temp vars to save y(X)
            y_X = y_X - np.mean(y_X, axis=0)  # centralized feature
            g_Y = pred_dummy[1 + self.u_size * 3 + self.y_size * 2 + ind_y]  # temp vars to save g(Y)
            g_Y = g_Y - np.mean(g_Y, axis=0)  # centralized feature

            py_X = py[ind_y] * (1 + np.matmul(y_X, g_Y.T))  # prob of each class
            p_accu.append(np.mean(np.argmax(py_X, axis=1) == np.argmax(y_labels[ind_y], axis=1)))  # calculate accu

        u_accu = []
        for ind_u in range(self.u_size):
            u_X = pred[1 + self.u_size + self.y_size + ind_u]
            u_X = u_X - np.mean(u_X, axis=0)

            f_U = pred_dummy[1 + self.u_size * 2 + self.y_size * 2 + ind_u]
            f_U = f_U - np.mean(f_U, axis=0)
            pu_X = pu[ind_u] * (1 + np.matmul(u_X, f_U.T))  # prob of each class
            u_accu.append(np.mean(np.argmax(pu_X, axis=1) == np.argmax(u_labels[ind_u], axis=1)))
        return 1 - np.mean(pred[0]), u_accu, p_accu

    def train(self):

        dict_val, _ = next(self.celeba_generator.nextVal())
        sample_input_x = dict_val['input_x']
        save_some(sample_input_x[:10], os.path.join(self.log_dir, 'original_image'))

        print("start training:")
        start_time = time.time()
        for j in range(self.start_epoch, self.num_epochs):
            # clean up loss logs on epoch begin, loss list different when u_size = 0
            if self.u_size > 0:
                loss_g = [[] for _ in range(4)]
                loss_d = [[] for _ in range(6)]
            else:
                loss_g = [[] for _ in range(3)]
                loss_d = [[] for _ in range(5)]
            loss_t = []
            for i in range(self.celeba_generator.steps_per_epoch):
                # start_1 = time.time()
                dict_input, output = next(self.celeba_generator.nextTrain())
                output = output.reshape(self.batch_size)
                input_x = dict_input['input_x']
                input_y = dict_input['input_y']
                input_u = dict_input['input_u']
                input_y_batch = []
                input_u_batch = []
                for ind_1 in range(self.y_size):
                    input_y_batch.append(input_y[ind_1])
                for ind_2 in range(self.u_size):
                    input_u_batch.append(input_u[ind_2])

                for _ in range(self.d_ites):
                    d_loss_list = self.d_train(
                        [input_x, np.random.uniform(0, 1, (self.batch_size, 1, 1, 1))] + input_u_batch + input_y_batch)

                errT = []
                for ind_t in range(self.y_size):
                    errT.append(self.transform_train[ind_t]([input_x] + input_u_batch + input_y_batch))

                for _ in range(self.g_ites):
                    g_loss_list = self.g_train([input_x] + input_u_batch + input_y_batch)

                if i % 100 == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    # trans_x = self.g_model.predict(input_x)
                    # input_u_dummy = [np.eye(2) for _ in range(self.u_size)]
                    # input_y_dummy = [np.eye(2) for _ in range(self.y_size)]
                    # pred_real = self.d_model.predict([input_x]+input_u_batch+input_y_batch)
                    # pred_trans = self.d_model.predict([trans_x]+input_u_batch+input_y_batch)
                    # pred_dummy = self.d_model.predict([input_x[:2]]+input_u_dummy+input_y_dummy)
                    # real_auth_score, real_uaccu, real_paccu = self.hscore_accu(pred_real,pred_dummy,[0.5 for _ in range(5)],[0.5],input_u_batch,input_y_batch)
                    # trans_auth_score, trans_uaccu, trans_paccu = self.hscore_accu(pred_trans,pred_dummy,[0.5 for _ in range(5)],[0.5],input_u_batch,input_y_batch)
                    if self.u_size > 0:
                        print(
                            'Elapsed [{}], epoch{}/{}, batch:{}/{},loss_real D:{},loss_fake D:{},loss_u D:{},loss_y D:{}, loss_gp D:{}, loss_t:{}'.
                            format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                                   np.mean(d_loss_list[0]), np.mean(d_loss_list[1]),
                                   np.mean(d_loss_list[2]), np.mean(d_loss_list[3]), np.mean(d_loss_list[4]),
                                   np.mean(errT)))
                        print('Elapsed [{}], epoch{}/{}, batch:{}/{},loss_real G:{},loss_u G:{},loss_y G:{}'.
                              format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                                     np.mean(g_loss_list[0]), np.mean(g_loss_list[1]),
                                     np.mean(g_loss_list[2])))
                        # print('Elapsed [{}], epoch{}/{}, batch:{}/{}, real_auth_score:{}, real_uaccu:{}, real_paccu:{}'.
                        #       format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                        #              real_auth_score,real_uaccu,real_paccu))
                        # print('Elapsed [{}], epoch{}/{}, batch:{}/{}, trans_auth_score:{}, trans_uaccu:{}, trans_paccu:{}'.
                        #       format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                        #              trans_auth_score, trans_uaccu, trans_paccu))
                    else:
                        print(
                            'Elapsed [{}], epoch{}/{},batch:{}/{},loss_real D:{},loss_fake D:{},loss_y D:{}, loss_gp D:{}, loss_t:{}'.
                            format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                                   np.mean(d_loss_list[0]), np.mean(d_loss_list[1]),
                                   np.mean(d_loss_list[2]), np.mean(d_loss_list[3]), np.mean(errT)))
                        print('Elapsed [{}], epoch{}/{},batch:{}/{},loss_real G:{},loss_y G:{}'.
                              format(et, j, self.num_epochs, i, self.celeba_generator.steps_per_epoch,
                                     np.mean(g_loss_list[0]), np.mean(g_loss_list[1])))

                    for ind_1 in range(len(d_loss_list)):
                        loss_d[ind_1].append(np.mean(d_loss_list[ind_1]))

                    for ind_2 in range(len(g_loss_list)):
                        loss_g[ind_2].append(np.mean(g_loss_list[ind_2]))

                    loss_t.append(np.mean(errT))

                if i % 1000 == 0:
                    val_trans_x = self.g_model.predict(sample_input_x)
                    save_some(val_trans_x[:10], os.path.join(self.log_dir, 'predict_' + str(j) + '_' + str(i)))

            # save g/d/t_models
            if (j + 1) % 5 == 0:
                self.g_model.save_weights(os.path.join(self.log_dir, 'g_model_' + self.date + '_' + str(j) + '.h5'))
                self.d_model.save_weights(os.path.join(self.log_dir, 'd_model_' + self.date + '_' + str(j) + '.h5'))

                for ind_y in range(self.y_size):
                    self.transform_tail_models[ind_y].save_weights(
                        os.path.join(self.log_dir, 't_model_' + self.date + '_' + str(j) + '_y_' + str(ind_y) + '.h5'))

            gan_result = {}
            gan_result['t_loss'] = loss_t
            if self.u_size > 0:
                gan_result['d_loss_real'] = loss_d[0]
                gan_result['d_loss_fake'] = loss_d[1]
                gan_result['d_loss_u'] = loss_d[2]
                gan_result['d_loss_y'] = loss_d[3]
                gan_result['gp'] = loss_d[4]
                gan_result['g_loss_real'] = loss_g[0]
                gan_result['g_loss_u'] = loss_g[1]
                gan_result['g_loss_y'] = loss_g[2]
            else:
                gan_result['d_loss_real'] = loss_d[0]
                gan_result['d_loss_fake'] = loss_d[1]
                gan_result['d_loss_y'] = loss_d[2]
                gan_result['gp'] = loss_d[3]
                gan_result['g_loss_real'] = loss_g[0]
                gan_result['g_loss_y'] = loss_g[1]

            np.save(os.path.join(self.log_dir, 'gan_result_' + self.date + '_' + str(j) + '.npy'), gan_result)


if __name__ == '__main__':
    import os
    from keras import backend as K
    from keras.backend.tensorflow_backend import set_session

    batch_size = 16
    K.set_learning_phase(False)
    set_session(tf.Session())
    for gamma in [0.1, 1, 10]:
        for ind_mode in [3]:
            log_dir = 'logs_mnist_evenfive_mode_' + str(ind_mode) + '_gamma_' + str(gamma)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            mnist_generator = reversedMNISTGenerator()
            privacy_net = privacyNetV3(log_dir=log_dir,
                                       g_model_dir=None,
                                       d_model_dir=None,
                                       start_epoch=0,
                                       batch_size=batch_size,
                                       image_size=(28, 28, 1),
                                       data_generator=mnist_generator,
                                       u_input_dims=[10],
                                       u_feat_dims=[10],
                                       y_feat_dims=[2],
                                       u_size=1,
                                       mode=ind_mode,
                                       epochs=10, lambda_cls=1,
                                       gamma=gamma, activation='tanh', t_ites=1)
            privacy_net.train()