import numpy as np

class Func:
    def step(self, x):
        if x > 0:
            return 1
        return 0;

    def ReLU(self, x):
        if x > 0:
            return x
        return 0
    
    def sigmoid(self, x, eita = 1.):
        return 1/(1+np.exp(-eita*x))

class Cell:
    def sigmoid(self, x, w, h = 1):
        return Func.sigmoid(np.inner(x,w) - h)
    def calc(self, x, w, h = 1):
        return np.inner(x,w) - h

def main():
    return
    
if __name__ == "__main__":
    main()
