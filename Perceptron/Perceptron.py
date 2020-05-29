import numpy as np
import matplotlib.pyplot as plt

PATH_X = "./../input_x.npy"
PATH_Y = "./../input_y.npy"

def to_input(data):
    x = data[0]
    y = data[1]
    n = x * 16 + y
    return np.array([int(k) for k in format(n, '08b')])


class Perceptron:
    def __init__(self, m, n, o):
        # decide initial weight [-0.005,0.005)
        self.w_IM = np.random.rand(n,m+1) - 0.5
        self.w_IM = self.w_IM / 100
        self.w_MO = np.random.rand(o,n+1) - 0.5
        self.w_MO = self.w_MO / 100

    def get_acc(self, x, y):
        ok = 0
        for i in range(len(x)):
            mid_in = np.inner(np.append(x[i],1.), self.w_IM)
            mid_out = np.array([int(k > 0) for k in mid_in])
            out_in = np.inner(np.append(mid_out,1.), self.w_MO)
            ok += int(int(out_in[0] > 0) == y[i])
        return ok / len(x)

    def learn(self, train_x, train_y, eta = 0.00001):
        mid_in = np.inner(np.append(train_x,1.), self.w_IM)
        mid_out = np.array([int(k > 0) for k in mid_in])
        out_in = np.inner(np.append(mid_out,1.), self.w_MO)
        out = int(out_in[0] > 0)
        self.w_MO[0,:-1] = self.w_MO[0,:-1] + ita * (train_y - out) * mid_out

def main():
    # read datas
    x = np.load(PATH_X)
    y = np.load(PATH_Y)
    # split datas
    train_x, test_x = np.split(x, 2)
    train_y, test_y = np.split(y, 2)
    # preprocess - transfer data into inputs
    datas = np.array([to_input(k) for k in train_x])
    tests = np.array([to_input(k) for k in test_x])
    # number of neurons input layer
    m = 8
    # number of neurons mid layer
    n = 10
    # number of neurons output layer
    o = 1
    # define the perceptron
    P = Perceptron(m,n,o)
    
    # learning time
    N = 10
    cnt = 0

    x = np.linspace(0,200,200)
    acc_train = np.copy(x)
    acc_test = np.copy(x)
    while True:
        acc = P.get_acc(datas, train_y)
        acc_train[cnt] = acc
        acc = P.get_acc(tests, test_y)
        acc_test[cnt] = acc
        print("Try ", cnt, ": ", acc)
        cnt += 1
        for i in range(len(datas)):
            P.learn(datas[i], train_y[i])
        if cnt >= 200:
            break
    plt.plot(x,acc_train,label="train")
    plt.plot(x,acc_test,label="test")
    plt.savefig("result.png")

if __name__ == "__main__":
    main()