import numpy as np
import yaml
import math
from pricing import Black_Scoles_pricing, binomial_option_pricing


def create_train_data(N, X, T, sigma, m, option_price, data_mode='normal steps', stopping_nodes = None, r=0): # TODO: add additional parameters to config file

    assert (data_mode in ['normal steps', 'discrete steps']), 'data mode in [normal steps, discrete steps]'

    gamma = 1.0
    grid = [(i / N) ** gamma * T for i in range(N + 1)]

    Ktrain = 10 ** 5
    initialprice = X

    # xtrain consists of the price S0,
    # the initial hedging being 0, and the increments of the log price process

    if stopping_nodes is None:
        stopping_steps = None
    else:
        stopping_steps = {}

    if data_mode == 'discrete steps':
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        q = (math.exp(r * dt) - d) / (u - d)

        increment = []
        u_counts = np.zeros(Ktrain)
        reg1 = np.ones(Ktrain)
        reg2 = np.zeros(Ktrain)
        for i in range(N):
            random = np.random.random(Ktrain)
            incr = np.array([(u - 1) if x < q else (d - 1) for x in random])
            u_count = np.array([1 if x < q else 0 for x in random])
            u_counts = u_counts + u_count
            u_counts = (u_counts * reg1) + reg2
            increment.append(incr.reshape(len(incr), 1))
            if (stopping_nodes is not None) & (i!=0):
                considered_up_movements = [x[1] for x in stopping_nodes if (x[0] == i)]
                if considered_up_movements == []:
                    continue
                else:
                    for index,count in enumerate(u_counts):
                        if count in considered_up_movements:
                            stopping_steps[index] = i
                            reg1[index] = 0
                            reg2[index] = -1
        if stopping_nodes is not None:
            stopping_steps[0] = N
            for i,x in enumerate(reg1):
                if x == 1:
                    stopping_steps[i] = N

    else:
        increment = [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
                                (Ktrain, m)) for i in range(N)]


    # x_train = ([initialprice * np.ones((Ktrain, m))] +
    #           [np.zeros((Ktrain, m))] +
    #           [np.ones((Ktrain, m))] +
    #           [option_price * np.ones((Ktrain, m))] +
    #           [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
    #                             (Ktrain, m)) for i in range(N)])
    x_train = ([initialprice * np.ones((Ktrain, m))] +
              [np.zeros((Ktrain, m))] +
              [np.ones((Ktrain, m))] +
              [option_price * np.ones((Ktrain, m))] +
              increment)

    y_train = np.zeros((Ktrain, 1 + N), dtype=np.float32)

    return x_train, y_train, stopping_steps


def create_test_data(N, X, T, sigma, m, option_price, data_mode='normal steps', stopping_nodes = None, r=0): # TODO: add additional parameters to config file

    assert (data_mode in ['normal steps', 'discrete steps']), 'data mode in [normal steps, discrete steps]'

    gamma = 1.0
    grid = [(i / N) ** gamma * T for i in range(N + 1)]

    Ktest = 50
    initialprice = X

    # xtrain consists of the price S0,
    # the initial hedging being 0, and the increments of the log price process

    if stopping_nodes is None:
        stopping_steps = None
    else:
        stopping_steps = {}

    if data_mode == 'discrete steps':
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        q = (math.exp(r * dt) - d) / (u - d)

        increment = []
        u_counts = np.zeros(Ktest)
        reg1 = np.ones(Ktest)
        reg2 = np.zeros(Ktest)
        for i in range(N):
            random = np.random.random(Ktest)
            incr = np.array([(u - 1) if x < q else (d - 1) for x in random])
            u_count = np.array([1 if x < q else 0 for x in random])
            u_counts = u_counts + u_count
            u_counts = (u_counts * reg1) + reg2
            increment.append(incr.reshape(len(incr), 1))
            if (stopping_nodes is not None) & (i!=0):
                considered_up_movements = [x[1] for x in stopping_nodes if (x[0] == i)]
                if considered_up_movements == []:
                    continue
                else:
                    for index,count in enumerate(u_counts):
                        if count in considered_up_movements:
                            stopping_steps[index] = i
                            reg1[index] = 0
                            reg2[index] = -1
        if stopping_nodes is not None:
            stopping_steps[0] = N
            for i,x in enumerate(reg1):
                if x == 1:
                    stopping_steps[i] = N
    else:
        increment = [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
                                (Ktest, m)) for i in range(N)]

    # x_test = ([initialprice * np.ones((Ktest, m))] +
    #          [np.zeros((Ktest, m))] +
    #          #[np.linspace(0.5, 1.5, Ktest)] +  # change this if you go to higher dimensions, for visualization purposses
    #          [np.linspace(0.5, 1.5, Ktest).reshape(Ktest, 1)] +  # change this if you go to higher dimensions, for visualization purposses
    #          [option_price * np.ones((Ktest, m))] +
    #          [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
    #                            (Ktest, m)) for i in range(N)])
    x_test = ([initialprice * np.ones((Ktest, m))] +
              [np.zeros((Ktest, m))] +
              # [np.linspace(0.5, 1.5, Ktest)] +  # change this if you go to higher dimensions, for visualization purposses
              [np.linspace(0.5, 1.5, Ktest).reshape(Ktest,
                                                    1)] +  # change this if you go to higher dimensions, for visualization purposses
              [option_price * np.ones((Ktest, m))] +
              increment)


    return x_test, stopping_steps
