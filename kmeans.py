import numpy as np
import time
from matplotlib import pyplot as plt

loop_time = 0
K_MIN = 2
K_MAX = 8
POINT_MIN = 50
POINT_MAX = 500
MAX_LOOP = 100

def k_means(data, k, max_time=MAX_LOOP):
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


def get_k_means(k, point_num):
    global loop_time
    start = time.time()
    if(k <= 8):
        color_box = ['y', 'm', 'c', 'r', 'g', 'b', 'w', 'k']
    else:
        color_box = range(k)
    for i in range(k):
        tmp_data = np.random.randn(point_num, 2) + i
        if(i == 0):
            data = tmp_data
        else:
            data = np.concatenate([data, tmp_data], 0)
    data.reshape(k, point_num, 2)
    color = k_means(data, k).tolist()
    color = [(lambda x: color_box[int(x)])(i) for i in color]
    plt.figure('result')
    plt.scatter(data[:, 0], data[:, 1], c=color)
    used_time = time.time() - start
    plt.title('result k={} in {:.3f}s with {} loop'.format(
        k, used_time, loop_time))
    plt.show()


if __name__ == '__main__':
    point_num = 100
    print('k means 分类随机点')
    while(1):
        k = input('请输入k的大小（{}-{}），或输入q退出：'.format(K_MIN, K_MAX))
        try:
            if(k == 'q'):
                break
            k = int(k)
        except:
            print('只允许输入数字和q，请重试。')
            continue
        if(k >= K_MIN and k <= K_MAX):
            while(1):
                point_num = input('请输入每组的点数({}-{})：'.format(POINT_MIN, POINT_MAX))
                try:
                    point_num = int(point_num)
                    if(point_num < POINT_MIN or point_num > POINT_MAX):
                        print('输入错误，只允许输入{}到{}的数字，请重新输入。'.format(POINT_MIN, POINT_MAX))
                        continue
                    break
                except:
                    print('输入错误，只允许输入{}到{}的数字，请重新输入。'.format(POINT_MIN, POINT_MAX))
            get_k_means(int(k), point_num)
        else:
            print('输入异常，请重新输入。')
    print('Bye')
