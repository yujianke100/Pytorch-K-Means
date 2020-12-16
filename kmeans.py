import numpy as np
import time
from matplotlib import pyplot as plt

BAR_LEN = 0.05
loop_time = 0


def k_means(data, k, max_time=100):
    global loop_time
    data_size, rgb = data.shape
    k_points = data[np.random.randint(0, data_size, k)]
    last_labels = 0
    for loop_time in range(max_time):
        matrx = np.expand_dims(data, 0).repeat(k, 0)
        k_points_matrx = np.expand_dims(k_points, 1).repeat(data_size, 1)
        distances = abs(matrx - k_points_matrx).sum(2)
        labels = distances.argmin(0)
        if((labels == last_labels).all()):
            break
        last_labels = labels
        for i in range(k):
            k_point = data[labels == i]
            if(i == 0):
                k_points = np.expand_dims(k_point.mean(0), 0)
            else:
                k_points = np.concatenate(
                    [k_points, np.expand_dims(k_point.mean(0), 0)], 0)
    return last_labels.astype(np.float64)


def get_k_means(k):
    global loop_time
    start = time.time()
    color_box = ['y', 'm', 'c', 'r', 'g', 'b', 'w', 'k']
    for i in range(k):
        tmp_data = np.random.randn(100, 2) + i
        if(i == 0):
            data = tmp_data
        else:
            data = np.concatenate([data, tmp_data], 0)
    data.reshape(k, 100, 2)
    color = k_means(data, k).tolist()
    color = [(lambda x: color_box[int(x)])(i) for i in color]
    plt.figure('result')
    plt.scatter(data[:, 0], data[:, 1], color=color)
    used_time = time.time() - start
    plt.title('result k={} in {:.3f}s with {} loop'.format(
        k, used_time, loop_time))
    plt.show()


if __name__ == '__main__':
    print('k means 分类随机点')
    while(1):
        k = input('请输入k的大小（2-8），或输入q退出：')
        if(k == 'q'):
            break
        elif(k >= '2' and k <= '8'):
            get_k_means(int(k))
        else:
            print('输入异常，请重新输入。')
    print('Bye')
