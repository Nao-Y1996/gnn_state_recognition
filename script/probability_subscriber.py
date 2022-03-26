#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import socket
import pickle
import os
import pandas as pd
from std_msgs.msg import Float32MultiArray


def show_probability_graph(ax, labels, probability, user_name):
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, probability, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.title('state pattern of ' + user_name)
    plt.ylim(0, 1)
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.draw()  # 描画する。
    plt.pause(0.01)  # 0.01 秒ストップする。
    plt.cla()


class ProbabilitySubscriber():
    def __init__(self):
        self.probability = []
        self.probability_sub = rospy.Subscriber("/avarage_probability", Float32MultiArray, self.callback)
        # rospy.wait_for_message("/avarage_probability", Float32MultiArray, timeout=10.0)

    def callback(self, data):
        self.probability = data.data

    def get_data(self):
        return self.probability

if __name__ == '__main__':

    rospy.init_node('probability_subscriber', anonymous=True)
    spin_rate=rospy.Rate(10)


    # 認識結果（probability）を受け取るための通信の設定
    probability_sub = ProbabilitySubscriber()



    # 認識の確率表示のグラフ設定
    def update_state_label(user_dir):
        labels = []
        read_data = pd.read_csv(user_dir +'/state.csv',encoding="utf-8")
        labels = read_data['state'].tolist()
        return labels

    user_name = rospy.get_param('user_name')
    user_dir = user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name
    labels = []
    
    fig, ax = plt.subplots()

    rospy.set_param("/robot_mode", "finish_train") # プログラム起動時に１度labelsを取得するための対応
    while not rospy.is_shutdown():
        robot_mode = rospy.get_param('robot_mode')

        if robot_mode=='finish_train':
            labels = update_state_label(user_dir)
            print(labels)
            rospy.set_param("/robot_mode", "nomal")
        elif robot_mode=='state_recognition':
            average_probability = probability_sub.get_data()
            # 未学習の状態パターンがあるときは、その状態は無視して認識を行う
            # （モデルの分類可能数とラベル数が一致しないときはラベルの末尾を削除する）
            if len(average_probability) != len(labels):
                labels = labels[0:-1]
                continue
            print(average_probability)
        else:
            average_probability = [0.0] * len(labels)
            pass

        # 認識確率の表示
        try:
            show_probability_graph(ax, labels, np.round(average_probability, decimals=4).tolist(), user_name)
        except ValueError:
            continue
        except NameError: # name 'average_probability' is not defined
            pass 
        

        spin_rate.sleep()