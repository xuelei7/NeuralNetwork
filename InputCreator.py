import numpy as np

PATH_X="input_x"
PATH_Y="input_y"

def make_input():
    N = 2000 # number of test cases
    d = 2 # number of dimensions

    x = np.random.randint(0, 16, (N,2)) # random numbers

    # result
    y = np.array([int(k[0] > k[1]) for k in x])

    np.save(PATH_X,x)
    np.save(PATH_Y,y)

def main():
    make_input()
    # test:
    xx = np.load(PATH_X+".npy")
    yy = np.load(PATH_Y+".npy")
    print(xx[0], yy[0])


if __name__ == "__main__":
    main()