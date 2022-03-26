#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from ipaddress import ip_address
import torch
import torch.nn as nn

from torch_scatter import scatter_max
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from graph_tools import graph_utilitys

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import random
import csv
import os
import glob
import sys
import shutil
import json
import time

class NNConvNet(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim):
        super(NNConvNet, self).__init__()
        self.edge_fc1 = nn.Linear(edge_feature_dim, node_feature_dim*32)
        self.nnconv1 = NNConv(node_feature_dim, 32, self.edge_fc1, aggr="mean")
        self.edge_fc2 = nn.Linear(edge_feature_dim, 32*48)
        self.nnconv2 = NNConv(32, 48, self.edge_fc2, aggr="mean")
        self.edge_fc3 = nn.Linear(edge_feature_dim, 48*64)
        self.nnconv3 = NNConv(48, 64, self.edge_fc3, aggr="mean")
        self.edge_fc4 = nn.Linear(edge_feature_dim, 64*128)
        self.nnconv4 = NNConv(64, 128, self.edge_fc4, aggr="mean")
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.nnconv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(model, iterator, optimizer, criterion):
    model.train()
    total_data_len = 0
    total_loss = 0
    correct_num = 0
    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        _, pred_labels = torch.max(pred, axis=1)
        correct_num += torch.sum(pred_labels == batch.y)
        total_data_len += len(pred_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss, epoch_accuracy = total_loss/total_data_len, float(correct_num)/total_data_len
    return epoch_loss, epoch_accuracy

def test(model, iterator):
    model.eval()
    total_data_len = 0
    total_loss = 0
    correct_num = 0
    for batch in iterator:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        _, pred_labels = torch.max(pred, axis=1)
        correct_num += torch.sum(pred_labels == batch.y)
        total_data_len += len(pred_labels)
        total_loss += loss.item()
    epoch_loss, epoch_accuracy = total_loss/total_data_len, float(correct_num)/total_data_len
    return epoch_loss, epoch_accuracy

if __name__=='__main__':
    ft_path = os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin'
    graph_utils = graph_utilitys(fasttext_model=ft_path)


    while not rospy.is_shutdown():

        robot_mode = rospy.get_param("/robot_mode")
        if robot_mode=='auto_train':

            # select user
            user_name = rospy.get_param("/user_name")


            # ------------------- データすべてをPositionData_4_trainに集約する -------------------
            user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name
            # PositionData_4_Trainの中身をリセットする
            shutil.rmtree(user_dir+'/PositionData_4_Train')
            os.mkdir(user_dir+'/PositionData_4_Train')
            # 収集したデータを読み込む
            experiment_dirs = glob.glob(user_dir+'/20*/')
            experiment_dirs.sort()
            for dir in experiment_dirs:
                files = glob.glob(dir + "position_data/pattern_*")
                files.sort()
                for data_file in files:
                    state_id = data_file.replace('.csv','')[-1]
                    with open(data_file, 'r') as f1:
                        csv_file = csv.reader(f1)
                        _data = [row for row in csv_file]
                    # PositionData_4_Train内のcsvファイルに書き込む
                    with open(user_dir+'/PositionData_4_Train/pattern'+state_id+'.csv', 'a') as f2:
                        writer = csv.writer(f2)
                        writer.writerows(_data)
            # -----------------------------------------------------------------------------------

            train_data_files = glob.glob(user_dir+'/PositionData_4_Train/pattern*')
            train_data_files.sort()
            pattern_num = len(train_data_files)
            print('number of state pattern : ', pattern_num)
            csv_path_dict = {}
            for i, file in enumerate(train_data_files):
                csv_path_dict[i] = file

            csv_path_dict_for_train = csv_path_dict

            # グラフデータセットを作成
            datasets,_ = graph_utils.csv2graphDataset(csv_path_dict_for_train)
            print("dataset length : ", len(datasets))

            # データセットをシャッフル
            random.shuffle(datasets)
            # データセットをtrainとtestに0.5:0.5で分割
            train_dataset = datasets[:int(len(datasets)*0.5)]
            test_dataset = datasets[int(len(datasets)*0.5):]
            
            # バッチサイズの決定
            if len(datasets) > 1000:
                # ミニバッチ学習
                batch_size = 1000
                model_name = 'Minibach_'+str(batch_size)
            else:
                # バッチ学習
                batch_size = len(datasets)
                model_name = 'Batchlearning'

            

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = NNConvNet(node_feature_dim=300, edge_feature_dim=3, output_dim=pattern_num).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())


            save_file = user_dir +'/learning_outputs/'+model_name+'.csv'
            with open(save_file,'w') as f:
                print( 'learning outputs is saved to ',save_file.split(user_name)[-1])

            print('-------train/test---------')
            total_epoch = 100
            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []
            for epoch in range(total_epoch):
                train_loss, train_acc = train(model, train_loader , optimizer, criterion)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                test_loss, test_acc = test(model, test_loader)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                print(f'epoch = {epoch+1}')
                print(f'train loss = {train_loss}  train Accuracy = {train_acc}')
                print(f'test loss = {test_loss}  test Accuracy = {test_acc}')
                with open(user_dir +'/learning_outputs/'+model_name+'.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, train_acc, test_acc,train_loss,test_loss])

            # モデルを保存(２箇所)
            model_path = user_dir + '/learning_outputs/'+model_name+'_nnconv.pt'
            torch.save(model.state_dict(), model_path)

            try:
                os.makedirs('/home/'+ os.getlogin()+'/catkin_ws/src/gnn_state_recognition/script/recognition_model')
            except OSError:
                pass
            model_path = '/home/'+ os.getlogin()+'/catkin_ws/src/gnn_state_recognition/script/recognition_model/'+user_name+'_model.pt'
            torch.save(model.state_dict(), model_path)

            # 現状の認識モデルの情報（user_name, model_name, pattern_num）を保存
            recognition_conf = {"model_name":model_name, "pattern_num":pattern_num}
            with open(user_dir+'/model_info.json', 'w') as f:
                json.dump(recognition_conf, f)
            with open('/home/'+ os.getlogin()+'/catkin_ws/src/gnn_state_recognition/script/recognition_model/'+user_name+'_model_info.json', 'w') as f:
                json.dump(recognition_conf, f)

            x = range(len(train_acc_list))
            # lossの描画
            fig = plt.figure()
            plt.plot(x, train_loss_list, color='b')
            plt.ylabel("Train Loss")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"LossTrain.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, test_loss_list, color='y')
            plt.ylabel("Test Loss")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"LossTest.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, train_loss_list, color='b', label='train')
            plt.plot(x, test_loss_list, color='y', label='test')
            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=15)
            plt.ylabel("Loss")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"Loss.png")
            plt.close()

            # accの描画
            fig = plt.figure()
            plt.plot(x, train_acc_list, color='b')
            plt.ylim(0.0, 1.0)
            plt.ylabel("Train Accuracy")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"AccuracyTrain.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, test_acc_list, color='y')
            plt.ylim(0.0, 1.0)
            plt.ylabel("Test Accuracy")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"AccuracyTest.png")
            plt.close()

            fig = plt.figure()
            plt.plot(x, train_acc_list, color='b', label='train')
            plt.plot(x, test_acc_list, color='y', label='test')
            plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=15)
            plt.ylim(0.0, 1.0)
            plt.ylabel("Accuracy")
            fig.savefig(user_dir+"/learning_outputs/"+model_name+"Accuracy.png")
            plt.close()

            
            rospy.set_param("/robot_mode", "finish_train")
            # ここで「学習が完了しました」と発話させたいが、
            # robot_toolsのインスタンスは１つしか起動できないので、それを一つのノードにして、それに発話内容を投げるような仕組みが必要
        else:
            pass