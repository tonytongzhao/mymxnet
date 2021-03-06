import numpy as np


def one_hot(labels):
    classes=np.unique(labels)
    n_classes=classes.size
    one_hot_labels=np.zeros(labels.shape+(n_classes, ))
    for c in classes:
        one_hot_labels[labels==c,c]=1
    return one_hot_labels

def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def tanh(x):
    return np.tanh(x)

def sigmoid_d(x):
    s=sigmoid(x)
    return s*(1-s)

def tanh_d(x):
    s=np.exp(2*x)
    return (s-1.0)/(s+1.0)

def relu(x):
    return np.maximum(0.0,x)

def relu_d(x):
    dx=np.zeros(x.shape)
    dx[x>=0]=1
    return dx

if __name__=='__main__':
    a=np.arange(10)
    print one_hot(a)
