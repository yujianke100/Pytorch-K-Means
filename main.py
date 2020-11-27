# -*- coding:utf-8 -*-
import easygui as t
from KMeans import *

MAX_K = 20
MIN_K = 2


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