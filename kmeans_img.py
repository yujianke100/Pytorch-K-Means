# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
import PySimpleGUI as sg
import easygui as t
from matplotlib import pyplot as plt

MAX_K = 20
MIN_K = 2

BAR_LEN = 0.05
loop_time = 0


def k_means(data, k, max_time=100):
    global loop_time
    data_size, rgb = data.shape
    k_points = data[np.random.randint(0, data_size, k)]
    last_labels = line_len = 0
    for loop_time in range(max_time):
        matrx = np.expand_dims(data, 0).repeat(k, 0)
        k_points_matrx = np.expand_dims(k_points, 1).repeat(data_size, 1)
        distances = abs(matrx - k_points_matrx).sum(2)
        labels = distances.argmin(0)
        difference = (labels != last_labels).sum()
        if(loop_time == 0):
            line_len = float(difference * BAR_LEN)
            layout = [[sg.Text('running...')],
                      [sg.ProgressBar(line_len, orientation='h', size=(
                          20, 20), key='progressbar')],
                      [sg.Cancel()]]
            window = sg.Window('please be waitting!', layout)
            progress_bar = window['progressbar']
        event, values = window.read(timeout=10)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return
        progress_bar.UpdateBar(max(line_len - float(difference), 0.))
        if(difference == 0):
            break
        last_labels = labels
        for i in range(k):
            k_point = data[labels == i]
            if(i == 0):
                k_points = np.expand_dims(k_point.mean(0), 0)
            else:
                k_points = np.concatenate(
                    [k_points, np.expand_dims(k_point.mean(0), 0)], 0)
    cmp_labels = np.expand_dims(labels, 0).repeat(rgb, 0).transpose(1, 0)
    result = np.zeros_like(data).astype(np.float64)
    for i in range(k):
        result += np.where(cmp_labels == i, k_points[i], 0)
    window.close()
    return result


def show(result, img_cv, k, used_time):
    global loop_time
    result_img_cv = result.astype(np.uint8)
    cv2.imwrite('./result/result_with_k{}.png'.format(k),
                cv2.cvtColor(result_img_cv, cv2.COLOR_RGB2BGR))

    plt.figure('result')
    plt.subplot(1, 2, 1)
    plt.imshow(img_cv)
    plt.title('original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(result_img_cv)
    plt.title('result k={} in {:.3f}s with {} loop'.format(
        k, used_time, loop_time))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./result/result_cmp.png')
    img = cv2.imread('./result/result_cmp.png')
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_k_means(img, k):
    start = time.time()
    img_cv = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    cul = img_cv.shape[0]
    row = img_cv.shape[1]
    data = img_cv.reshape(cul * row, 3)

    result = k_means(data, k)

    if(result.all() == None):
        return
    result = result.reshape(cul, row, 3)
    used_time = time.time() - start

    show(result, img_cv, k, used_time)


def uiShow():
    supports = ['.jpg', '.jpeg', '.png', '.jfif']
    while (1):
        choise = t.ccbox(msg='K_means_demo',
                         title='K_means_demo', choices=('选择图片', '退出'))
        if(choise == False):
            break
        img_path = t.fileopenbox()
        if(img_path == None):
            t.msgbox('请选择图片！', 'error!', '重试')
            continue
        elif(img_path[-4:] not in supports and img_path[-5:] not in supports):
            t.msgbox('只支持jpg,jpeg,png,jfif文件！', 'error!', '重试')
            continue
        k = t.integerbox(msg='请输入K的大小({}-{})'.format(MIN_K, MAX_K), title='输入K', default='2',
                         lowerbound=MIN_K, upperbound=MAX_K, image=None, root=None,)
        if(k == None):
            continue
        while(k > MAX_K or k < MIN_K):
            t.msgbox('请选择图片！', 'error!', '重试')
            k = t.integerbox(msg='请输入K的大小({}-{})'.format(MIN_K, MAX_K), title='输入K', default='2',
                             lowerbound=MIN_K, upperbound=MAX_K, image=None, root=None,)
            if(k == None):
                continue
        if(k == None):
            continue
        get_k_means(img_path, k)


if __name__ == '__main__':
    uiShow()
