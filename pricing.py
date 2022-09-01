import math
import numpy as np
import scipy.stats as scipy

def binomial_option_pricing(S0, K, T, N, r=0, u=None, d=None, sigma=None, option_type='european call'):
    # http://www.josephthurman.com/binomial3.html
    # https://www.investopedia.com/articles/investing/021215/examples-understand-binomial-option-pricing-model.asp
    '''
    X: current stock price
    K: strike price
    T: maturity time in years
    N: time discretization
    r: annual risk-free rate of returns
    u: expected price increase
    d: expected price decrease
    sigma: volatility
    option_type: 'european' or 'american'
    '''
    assert ((sigma == None) or ((u == None) and (d == None))), 'Either sigma or (u,d) must be given, not both'
    assert ((sigma is not None) or (u is not None) or (d is not None)), 'Either sigma or (u,d) must be given'
    assert option_type in ['european call', 'european put',
                           'american call', 'american put'], 'Invalid option type'
    dt = T/N

    if sigma != None:
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u

    q = (math.exp(r * dt) - d) / (u - d)
    C = {} # option prices

    if option_type in ['european call', 'american call']:
        factor = 1
    else:
        factor = -1

    for m in range(0, N + 1):
        C[(N, m)] = max((S0 * (u ** (m)) * (d ** (N - m)) - K) * factor, 0)
    for k in range(N - 1, -1, -1): # iterate through timesteps
        for m in range(0, k + 1): # iterate through possible up-down movements
            edv = q * C[(k + 1, m + 1)] + (1 - q) * C[(k + 1, m)] # expected discounted value
            if option_type in ['european call', 'european put']:
                C[(k, m)] = math.exp(-r * dt) * edv
            elif option_type in ['american call', 'american put']:
                stock_price = S0 * (u ** (m)) * (d ** (k - m))
                intrinsic_value = (stock_price - K) * factor
                C[(k, m)] = max(math.exp(-r * dt) * edv, intrinsic_value) # since american option can be exercised before maturity

    return C[(0, 0)] # return calculated option price for the initial day



def Black_Scoles_pricing(S0, K, T, sigma):
    # code from lecture4 deep_hedging_bs_conditionalrewards
    '''
    S0: initial price
    K: strike price
    T: maturity
    sigma: volatility
    '''
    return S0*scipy.norm.cdf((np.log(S0/K)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-K*scipy.norm.cdf((np.log(S0/K)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))
