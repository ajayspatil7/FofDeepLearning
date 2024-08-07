import math


def step(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))


def tanH(x):
    return (math.exp(2 * x) - 1) / ( math.exp(2 * x) + 1)


def reLU(x):
    return max(0, x)


def softmax(x: list):
    exps = [math.exp(x) for i in x]
    exps_sum = sum(exps)
    prob = [j / exps_sum for j in exps]
    return prob