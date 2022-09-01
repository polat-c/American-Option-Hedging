import os
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Concatenate, Dropout, Subtract, \
                        Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot
from keras.backend import constant
from keras import optimizers

from keras.engine.topology import Layer
import keras
from keras.models import Model
from keras.layers import Input
from keras import initializers
from keras.constraints import max_norm
import keras.backend as K

import matplotlib.pyplot as plt

import yaml

from pricing import binomial_option_pricing, Black_Scoles_pricing
from loss import custom_loss, custom_loss_v2
from data import create_train_data, create_test_data
from utils import progress_bar, compute_stoppings


class Hedge():

    def __init__(self, args):

        self.config_file = args.config_file
        with open(self.config_file) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)

        # TODO: add model parameters (optimizer etc.)

        self.run_name = self.configs['run_name']
        self.ckpt_name = self.configs['ckpt_name']
        self.instrument = self.configs['instrument'] # at the moment, only european call
        self.superhedge = self.configs['superhedge']
        # TODO: instead of using N for binomial pricing, define a new discretization M ---> train N*M networks total (N for every possible exercise date)
        self.early_exercise = self.configs['early_exercise']
        assert (((self.instrument.split()[0] == 'european') and (self.superhedge == False)) or (self.instrument.split()[0] == 'american')), 'Cannot superhedge for european options'

        self.N = self.configs['N'] # time discretization
        self.X = self.configs['X'] # initial value of the asset
        self.T = self.configs['T'] # maturity
        self.K = self.configs['K'] # strike
        self.sigma = self.configs['sigma'] # volatility
        self.data_mode = self.configs['data_mode'] # data creation mode: either 'normal steps' or 'discrete steps'

        self.m = self.configs['m'] # dimension of price
        self.d = self.configs['d'] # number of layers in strategy
        self.n = self.configs['n'] # nodes in the first but last layers

        self.epochs = self.configs['epochs']
        self.batch_size = self.configs['batch_size']
        self.save_epoch = self.configs['save_epoch'] # save weights every 'save_epochs' epochs


        # PRICING
        if self.instrument == 'european call':
            self.option_price = Black_Scoles_pricing(self.X, self.K, self.T, self.sigma)
        else:
            self.option_price = binomial_option_pricing(self.X, self.K, self.T, self.N,
                                                   sigma=self.sigma, option_type=self.instrument)


        # DATA CREATION
        if (self.instrument == 'american put') & (self.early_exercise):
            assert (self.data_mode == 'discrete steps'), 'if early exercise is enabled, data has to be created using discrete steps mode'
            stopping_nodes = compute_stoppings(self.X, self.K, self.T, self.N, self.sigma)

            self.x_train, self.y_train, self.stopping_steps_train = create_train_data(self.N, self.X, self.T, self.sigma, self.m, self.option_price,
                                                                                      data_mode='discrete steps', stopping_nodes = stopping_nodes)
            self.x_test, self.stopping_steps_test = create_test_data(self.N, self.X, self.T, self.sigma, self.m, self.option_price,
                                                                      data_mode='discrete steps', stopping_nodes = stopping_nodes)
            self.sorted_node_list = []
            for index in sorted(self.stopping_steps_train):
                self.sorted_node_list.append([index, self.stopping_steps_train[index]])
            self.sorted_node_list = np.array(self.sorted_node_list)

        else:
            self.x_train, self.y_train, _ = create_train_data(self.N, self.X, self.T, self.sigma, self.m,
                                                                                      self.option_price)
            self.x_test, _ = create_test_data(self.N, self.X, self.T, self.sigma, self.m,
                                                                      self.option_price)


        # MODEL
        self.model_hedge_strat = self.create_strat()


        # LOAD / PREPARE CHECKPOINTS #
        run_path = os.path.join('./checkpoints', self.run_name)
        if self.ckpt_name == None:
            if not os.path.exists(run_path):
                os.makedirs(run_path, exist_ok=True)
            self.checkpoint_path = os.path.join('./checkpoints', self.run_name, 'cp.ckpt')
        else:
            self.checkpoint_path = os.path.join('./checkpoints', self.run_name, self.ckpt_name)
            self.model_hedge_strat.load_weights(self.checkpoint_path)
        ############################





    def train(self): # TODO: ultimately, get x_train, y_train from csv file

        self.model_hedge_strat.compile(optimizer='adam', loss=custom_loss)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_freq=5 * int(np.ceil(np.array(self.x_train).shape[1] / self.batch_size)))

        self.model_hedge_strat.fit(x=self.x_train, y=self.y_train, epochs=self.epochs, verbose=True, batch_size=self.batch_size, callbacks=[cp_callback])



    def manual_train_early_exercise(self):

        self.optimizer = keras.optimizers.Adam()  # TODO: add optimizer hyperparams

        self.x_train = np.array(self.x_train)
        self.x_train = np.transpose(self.x_train,
                                    (1, 0, 2))  # reshaping to make it compatible with keras dataset object
        self.x_test = np.array(self.x_test)
        self.x_test = np.transpose(self.x_test, (1, 0, 2))

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train, self.sorted_node_list))
        train_dataset = train_dataset.batch(self.batch_size)

        for epoch in range(self.epochs):
            print("\nEpoch %d" % (epoch + 1,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train, sorted_node_list_batch) in enumerate(train_dataset):
                #sorted_node_list_batch = [x[1] for x in sorted_node_list_batch]
                sorted_node_list_batch = [[i, x[1]] for i, x in enumerate(sorted_node_list_batch)]
                x_batch_train = np.transpose(x_batch_train, (1, 0, 2))
                x_batch_train = list(x_batch_train)

                with tf.GradientTape() as tape:
                    out = self.model_hedge_strat(x_batch_train, training=True)  # Logits for this minibatch
                    self.loss_value = custom_loss_v2(y_batch_train, out, sorted_node_list=sorted_node_list_batch)

                grads = tape.gradient(self.loss_value, self.model_hedge_strat.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model_hedge_strat.trainable_weights))

                progress_bar(step, len(train_dataset), 'Loss: {l}'.format(l=self.loss_value))

            self.model_hedge_strat.save_weights(self.checkpoint_path)



    def manual_train(self):

        self.optimizer = keras.optimizers.Adam() # TODO: add optimizer hyperparams

        self.x_train = np.array(self.x_train)
        self.x_train = np.transpose(self.x_train, (1,0,2)) # reshaping to make it compatible with keras dataset object
        self.x_test = np.array(self.x_test)
        self.x_test = np.transpose(self.x_test, (1, 0, 2))

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_dataset = train_dataset.batch(self.batch_size)

        # Freezing all weights
        for layer in self.model_hedge_strat.layers:
            layer.trainable = False
        self.model_hedge_strat.get_layer('premium').trainable = True

        for epoch in range(self.epochs):
            print("\nEpoch %d" % (epoch+1,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                x_batch_train = np.transpose(x_batch_train, (1,0,2))
                x_batch_train = list(x_batch_train)

                self.superhedge_loop(x_batch_train, y_batch_train)

                progress_bar(step, len(train_dataset), 'Loss: {l}'.format(l=self.loss_value))

            self.model_hedge_strat.save_weights(self.checkpoint_path)



    def test(self):

        y_test = self.model_hedge_strat.predict(self.x_test)
        return y_test


    def create_strat(self):

        # in case of superhedge, output dimension is (None, 2N + 1)
        if self.superhedge:
            sh_outputs = []

        price = Input(shape=(self.m,))
        hedge = Input(shape=(self.m,))
        hedgeeval = Input(shape=(self.m,))
        premium = Input(shape=(self.m,))

        inputs = [price] + [hedge] + [hedgeeval] + [premium]
        outputhelper = []

        premium = Dense(self.m, activation='linear', trainable=True,
                        kernel_initializer=initializers.RandomNormal(0, 1),  # kernel_initializer='random_normal',
                        bias_initializer=initializers.RandomNormal(0, 1), name='premium')(premium)

        layers = self.create_layers()

        for j in range(self.N):
            strategy = price
            strategyeval = hedgeeval
            for k in range(self.d):
                strategy = layers[k + (j) * self.d](
                    strategy)  # strategy at j is the hedging strategy at j , i.e. the neural network g_j
                strategyeval = layers[k + (j) * self.d](strategyeval)
            incr = Input(shape=(self.m,))
            logprice = Lambda(lambda x: K.log(x))(price)
            logprice = Add()([logprice, incr])
            pricenew = Lambda(lambda x: K.exp(x))(logprice)  # creating the price at time j+1
            priceincr = Subtract()([pricenew, price])
            hedgenew = Multiply()([strategy, priceincr])
            # mult = Lambda(lambda x : K.sum(x,axis=1))(mult) # this is only used for m > 1
            hedge = Add()([hedge, hedgenew])  # building up the discretized stochastic integral
            inputs = inputs + [incr]
            outputhelper = outputhelper + [strategyeval]
            price = pricenew

            ######### New output shape for superhedge ##########
            if self.superhedge:
                payoff = Lambda(lambda x: 0.5 * (K.abs(x - self.K) + x - self.K))(price)
                outputs = Subtract()([payoff, hedge])
                outputs = Subtract()([outputs, premium])  # payoff minus price minus hedge
                sh_outputs = sh_outputs + [outputs]

        if self.superhedge:

            sh_outputs = sh_outputs + outputhelper + [premium]
            sh_outputs = Concatenate()(sh_outputs)

            model_hedge_strat = Model(inputs=inputs, outputs=sh_outputs)

            return model_hedge_strat
        ##########################################################

        payoff = Lambda(lambda x: 0.5 * (K.abs(x - self.K) + x - self.K))(price)
        outputs = Subtract()([payoff, hedge])
        outputs = Subtract()([outputs, premium])  # payoff minus price minus hedge
        outputs = [outputs] + outputhelper + [premium]
        outputs = Concatenate()(outputs)

        model_hedge_strat = Model(inputs=inputs, outputs=outputs)

        return model_hedge_strat


    def superhedge_loop(self, x_batch_train, y_batch_train):
        '''
        unfreezes and refreezes the layers iteratively in order to train a single network for
        every discrete timepoint
        '''

        for n in range(self.N):

            for i in range(self.d):
                self.model_hedge_strat.get_layer(str(i) +'_'+ str(n)).trainable = True

            with tf.GradientTape() as tape:

                out = self.model_hedge_strat(x_batch_train, training=True)  # Logits for this minibatch
                self.loss_value = custom_loss(y_batch_train, out, index=n)

            grads = tape.gradient(self.loss_value, self.model_hedge_strat.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model_hedge_strat.trainable_weights))

            for i in range(self.d):
                self.model_hedge_strat.get_layer(str(i) +'_'+ str(n)).trainable = False



    def create_layers(self):

        layers = []
        for j in range(self.N):
            for i in range(self.d):
                if i < self.d - 1:
                    nodes = self.n
                    layer = Dense(nodes, activation='tanh', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 1),
                                  # kernel_initializer='random_normal',
                                  bias_initializer='random_normal',
                                  name=str(i) +'_'+ str(j))
                else:
                    nodes = self.m
                    layer = Dense(nodes, activation='linear', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 0.1),
                                  # kernel_initializer='random_normal',
                                  bias_initializer='random_normal',
                                  name=str(i) +'_'+ str(j))
                layers = layers + [layer]

        return layers
