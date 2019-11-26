import numpy as np


def match_rate(truth: np.ndarray, estim: np.ndarray) -> float:
    '''
    rate = match_rate(truth, estim)

    Params:
    truth: np.array()
    estim: np.array()

    Returns:
    dst: match rate
    '''
    upper = np.max([truth, estim], 0)
    lower = np.min([truth, estim], 0)
    dst = lower.sum() / upper.sum()
    return dst

if __name__ == '__main__':
    truth = np.random.randint(9, size=10) + 1
    estim = np.random.randint(9, size=10) + 1
    mr = match_rate(truth, estim)
    print('truth:', ['%.2f' % x for x in truth])
    print('estim:', ['%.2f' % x for x in estim])
    print('MR:', mr)
