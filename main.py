import math
import argparse
import numpy as np
import pickle
import scipy.stats as scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

from pricing import binomial_option_pricing, Black_Scoles_pricing
from hedging import Hedge
from data import create_train_data


def main(args):

    solver = Hedge(args)

    # solver.train()
    #
    # y = solver.test()
    #
    # return y

    #solver.train()
    #solver.manual_train()

    # TODO: uncomment to train
    #solver.manual_train_early_exercise()
    #solver.manual_train()

    x_test = solver.x_test
    y_test = solver.test()
    #cd
    with open('exp3_xtest.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(x_test, f, pickle.HIGHEST_PROTOCOL)
    with open('exp3_ytest.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='toy-AE')

    parser.add_argument('--config_file', default='./configs/example_config.yml', type=str, help='Path to config file')

    args = parser.parse_args()

    main(args)