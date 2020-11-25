import torch
import cv2
import numpy as np
import time
import PySimpleGUI as sg
from matplotlib import pyplot as plt


def k_means(data, k, max_time=100):
    size, rgb = data.shape
    init = torch.randint(size, (k,))
    k_points = data[init]
    last_label = line_len = 0
    for time in range(max_time):
        matrx = data.unsqueeze(0).repeat(k, 1, 1)
        k_points_matrx = k_points.unsqueeze(1).repeat(1, size, 1)
        distances = abs(matrx - k_points_matrx).mean(dim=2)
        label = distances.argmin(dim=0)
        difference = (label != last_label).sum()
        if(time == 0):
            line_len = difference
            layout = [[sg.Text('running...')],
                      [sg.ProgressBar(float(line_len), orientation='h', size=(
                          20, 20), key='progressbar')],
                      [sg.Cancel()]]
            window = sg.Window('please be waitting!', layout)
            progress_bar = window['progressbar']
        event, values = window.read(timeout=10)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return -1
        progress_bar.UpdateBar(float(line_len - difference))
        if(difference == 0):
            break
        last_label = label
        for i in range(k):
            k_point = data[label == i]
            if(i == 0):
                k_points = k_point.mean(0).unsqueeze(0)
            else:
                k_points = torch.cat(
                    [k_points, k_point.mean(0).unsqueeze(0)], 0)
    cmp_labels = label.expand(3, label.shape[0]).transpose(0, 1)
    result_list = []
    cal_data = torch.where(data == 0., data.type(torch.DoubleTensor), 1e-9)
    for i in range(k):
        tmp_data = torch.where(
            cmp_labels == i, data.type(torch.DoubleTensor), 0.)
        ave_r = tmp_data[:, 0].sum() / len(torch.nonzero(tmp_data[:, 0]))
        ave_g = tmp_data[:, 1].sum() / len(torch.nonzero(tmp_data[:, 1]))
        ave_b = tmp_data[:, 2].sum() / len(torch.nonzero(tmp_data[:, 2]))
        ave = torch.cat(
            (ave_r.unsqueeze(0), ave_g.unsqueeze(0), ave_b.unsqueeze(0)))
        tmp_data = torch.where(cmp_labels == i, ave, 0.)
        result_list.append(tmp_data)
    result = torch.zeros_like(data)
    for i in result_list:
        result += i
    window.close()
    return result


def get_k_means(img, k):
    start = time.time()
    img_cv = cv2.imread(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    tensor_cv = torch.from_numpy(img_cv).type(torch.FloatTensor)
    cul = tensor_cv.shape[0]
    row = tensor_cv.shape[1]
    data = tensor_cv.reshape(cul * row, 3)
    result = k_means(data, k)
    if(type(result) == int and result == -1):
        return
    result = result.reshape(cul, row, 3)
    result_img_cv = result.numpy().astype(np.uint8)
    cv2.imwrite('result.png', result_img_cv)
    used_time = time.time() - start
    
    plt.figure('result')
    plt.subplot(1,2,1)
    plt.imshow(img_cv)
    plt.title('original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,2,2)
    plt.imshow(result_img_cv)
    plt.title('result with k = {} in time {:.3f}s'.format(k, used_time))
    plt.xticks([])
    plt.yticks([])
    plt.ion()
    # plt.show()
    plt.savefig('result_cmp.png')
    img=cv2.imread('result_cmp.png')
    cv2.imshow("result", img)
    cv2.waitKey(0)    
    cv2.destroyAllWindows() 
    return
    