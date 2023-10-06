import numpy as np

if __name__ == '__main__':
    dataset = 'small'
    points_file = f'dataset/{dataset}_data.csv'
    labels_file = f'dataset/{dataset}_label.csv'

    file = open(points_file, 'r')
    lines = file.readlines()
    X = [[float(ele) for ele in line.split()] for line in lines[1:]]
    X = np.array(X)
    file.close()

    file = open(labels_file, 'r')
    lines = file.readlines()
    y = [float(line) for line in lines[1:]]
    y = np.array(y)
    file.close()

    n, m = X.shape

    denominators = np.array([X[:,i] @ X[:,i] for i in range(m)])

    w = np.zeros(m)

    iterations = 10
    for iters in range(iterations):
        #w_temp = np.zeros(m)

        for i in range(m):
            Xw_mi = X@w - X[:,i]*w[i]
            numerator = X[:,i] @ (y - Xw_mi)
            w[i] = numerator / denominators[i]

        #w[:] = w_temp

        diff = X@w - y
        print(diff @ diff)
