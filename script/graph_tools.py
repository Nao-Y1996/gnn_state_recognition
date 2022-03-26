#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import csv
import os
import itertools
from sklearn.preprocessing import minmax_scale
import fasttext
import fasttext.util
# fasttext.util.download_model('en', if_exists='ignore')
class DictConstrustionError(Exception):
    pass
import json

class graph_utilitys():
    def __init__(self, fasttext_model):
        # self.ID_2_OBJECT_NAME = {
        #             0:"face", 1:"bottle", 2:"wine glass", 3:"cup", 4:"fork", 5:"knife", 6:"spoon", 7:"bowl",
        #             8:"banana", 9:"apple", 10:"sandwich", 11:"orange", 12:"broccoli", 13:"carrot", 14:"hot dog", 15:"pizza", 16:"donut",
        #             17:"cake", 18:"chair", 19:"sofa", 20:"pottedplant", 21:"bed", 22:"diningtable", 23:"toilet", 24:"tvmonitor", 25:"laptop",
        #             26:"mouse", 27:"remote", 28:"keyboard", 29:"cell phone", 30:"microwave", 31:"oven", 32:"toaster", 33:"sink", 34:"refrigerator",
        #             35:"book", 36:"clock", 37:"vase", 38:"scissors", 39:"teddy bear", 40:"hair drier", 41:"toothbrush"
        #             }
        self.ID_2_OBJECT_NAME = {}
        conf_dir = os.path.dirname(__file__)+'/obj_conf/'
        with open(conf_dir+'ID_2_OBJECT_NAME.json') as f:
            _id2obj = json.load(f)
            for _id, _name in _id2obj.items():
                self.ID_2_OBJECT_NAME[int(_id)] = _name

        self.OBJECT_NAME_2_ID = {}
        conf_dir = os.path.dirname(__file__)+'/obj_conf/'
        with open(conf_dir+'OBJECT_NAME_2_ID.json') as f:
            _obj2id = json.load(f)
            for _name, _id in _obj2id.items():
                self.OBJECT_NAME_2_ID[_name] = int(_id)
        
        self.ft = fasttext.load_model(fasttext_model)

    def changeID_2_OBJECT_NAME(self, obj_name_changer_dict):
        """
        if change pattern is bellow
        'sandwich'-->'toast', 'robot'-->'camera'
        input would be
        obj_name_changer_dict = {'sandwich':'toast','robot':'camera'}
        """
        for origin_obj_name in obj_name_changer_dict.keys():
            # 変更したい物体名がID_2_OBJECT_NAMEのvalueに存在するかチェック
            if origin_obj_name in self.ID_2_OBJECT_NAME.values():
                # 変更したい名前のkey(id)をID_2_OBJECT_NAMEから見つける
                ids = [k for k, v in self.ID_2_OBJECT_NAME.items() if v == origin_obj_name]
                if len(ids) == 1:
                    # 物体名を変更する
                    id = ids[0]
                    self.ID_2_OBJECT_NAME[id] = obj_name_changer_dict[origin_obj_name]
                else:
                    raise DictConstrustionError('変更したい物体名のidが、ID_2_OBJECT_NAME内に複数見つかりました')
            else:
                raise DictConstrustionError('変更したい物体名が、ID_2_OBJECT_NAME内に見つかりません。')
    
    def removeDataId(self, data):
        position_data = data[1:]
        return position_data

    def positionData2graph(self, position_data, label, include_names=False):
        # 欠損値のあるデータはグラフに変換しない
        if None in position_data:
            return None, None
        obj_num = int(len(position_data)/4)
        #物体の数が0 or 1の時はグラフに変換しない
        if (obj_num==0) or (obj_num==1):
            return None, None
        names = []
        nodes_features = []
        positions = []
        for obj in np.array(position_data).reshape(-1,4):
            obj_id = obj[0]
            name = self.ID_2_OBJECT_NAME[obj_id]
            try:
                word_v = self.ft.get_word_vector(name).tolist()
            except KeyError:
                continue
            if include_names:
                names.append(name)
            nodes_features.append(word_v)
            x, y, z = obj[1], obj[2], obj[3]
            positions.append([x, y, z])
        x = torch.tensor(nodes_features,  dtype=torch.float)
        # print('nodes shape : \n',x.shape)
        # print('positions : \n', np.array(positions))

        # calculate distanse obj2obj
        position_dist_matrix = [[0 for i in range(obj_num)] for j in range(obj_num)]
        position_vector_matrix = [[0 for i in range(obj_num)] for j in range(obj_num)]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                vec = np.array(pos1) - np.array(pos2)
                position_vector_matrix[i][j] = vec
                dist =((np.linalg.norm(vec)).tolist())
                position_dist_matrix[i][j] = dist
        # position_normarized_dist_matrix = np.reshape(minmax_scale(np.array(position_dist_matrix).flatten()), (obj_num,obj_num))
        # print('position_vector_matrix : \n',np.array(position_vector_matrix))
        # print('position_dist_matrix : \n',np.array(position_dist_matrix))
        # print('position_normarized_dist_matrix : \n',position_normarized_dist_matrix)

        # create edge_index, edge_feature
        edges = []
        edge_features = []
        for i, _ in enumerate(position_dist_matrix):
            for j, _ in enumerate(position_dist_matrix):
                # dist = position_dist_matrix[i][j]
                # normarized_dist = position_normarized_dist_matrix[i][j]
                vec = position_vector_matrix[i][j]
                # ------------ ここのif文にedgeを作る条件を入れる ------------
                if i!=j: # 自己ループはなし
                    
                    # 基準(camera, face等)と物体は必ず接続、物体同士は0.3m以内であれば接続、エッジの特徴量は物体館の距離（距離は0~1正規化されたもの）
                    # if i==0 or j==0:
                    #     edges.append([i,j])
                    #     edge_features.append([normarized_dist])
                    #     continue
                    # if dist <=0.3:
                    #     edges.append([i,j])
                    #     edge_features.append([normarized_dist])
                    
                    # 全て接続、エッジの特徴量は物体同士の位置ベクトル
                    edges.append([i,j])
                    edge_features.append(vec)

        edge_index = torch.tensor(np.array(edges).T.tolist(), dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        graph = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_attr)
        return graph, names
    
    def convertData2dummygraphs(self, data):
        position_data = self.removeDataId(data)
        obj_num = len(position_data)/4
        dummy_graph_lsit = []
        removed_obj_data_list = []
        for i in range(int(obj_num)):
            removed_obj_data_list.append([position_data[i*4],position_data[i*4+1],position_data[i*4+2],position_data[i*4+3]])
            dummy_postion_data =  np.reshape(np.delete(np.reshape(position_data,(-1,4)), i, axis=0), (1,-1))[0]
            dummy_graph, dummy_names = self.positionData2graph(dummy_postion_data, label=12345, include_names=False)
            dummy_graph_lsit.append([dummy_graph, dummy_names])
        return dummy_graph_lsit, removed_obj_data_list

    def csv2graphDataset(self, csv_files, include_names=False):
        """
        csv_files = {0:'pattern0.csv', 1:'pattern1.csv',
                     2:'pattern2.csv', 3:'pattern3.csv'}
        """
        obj_names_sets = []
        datasets = []
        for num in range(len(csv_files)):
            file_path = csv_files[num]
            all_data = []
            with open(file_path) as f:
                csv_file = csv.reader(f)
                for row in csv_file:
                    # _row = []
                    # if '' in row:
                    #         continue
                    # for v in row:
                    #     _row.append(float(v))
                    _row = [float(v) for v in row if not '' in row]
                    all_data.append(_row)
            print(file_path.split('/')[-1],' number of data ---> ', len(all_data))
            for row, data in enumerate(all_data):
                position_data = self.removeDataId(data)
                graph, obj_names = self.positionData2graph(position_data, label=num, include_names=include_names)
                if graph is not None:
                    datasets.append(graph)
                    if len(obj_names)!=0:
                        obj_names_sets.append(obj_names)
        return datasets, obj_names_sets
    
    def CreateAugmentedData(self, data, remove_obj_id_list):
        AugmentedDataList = [data]

        data_id = data[0]
        position_data = np.reshape(data[1:], (-1,4))

        remove_pattern = []
        for i in range(1, len(remove_obj_id_list)+1):
            combinations =  list(itertools.combinations(remove_obj_id_list, i))
            for comb in combinations:
                remove_pattern.append(list(comb))
        # print(remove_pattern)
        for pattern in remove_pattern:
            # print(f'--------- パターン : {pattern} ---------')
            _position_data = position_data
            # print(_position_data)
            flag = False # パターンに含まれるid全てが、position_dataに含まれるかどうか
            for id in pattern:
                # print(f'id = {id} の行を削除')
                try:
                    remove_index = _position_data[:,0].tolist().index(id)
                    flag = True
                except ValueError:
                    # print(f'id = {id} の物体は含まれていませんでした')
                    flag = False
                    continue
                # print(f'{remove_index} 行目を削除')
                _position_data = np.delete(_position_data, obj=remove_index, axis=0)
            if flag and (_position_data.tolist() != position_data.tolist()):
                new_data = _position_data.flatten().tolist()
                new_data.insert(0, data_id)
                AugmentedDataList.append(new_data)
                # print(f'保存 : {new_data}')
            else:
                pass
        return AugmentedDataList

    def CreateAugmentedCSVdata(self, origin_csv, augmented_csv, remove_obj_id_list):
        with open(augmented_csv, 'w') as f:
            pass
        with open(origin_csv) as f:
            csv_file = csv.reader(f)
            data_num = 0
            for i, row in enumerate(csv_file):
                _row = []
                if '' in row:
                        continue
                for v in row:
                    _row.append(float(v))
                AugmentedDataList = self.CreateAugmentedData(_row, remove_obj_id_list)
                for Augmenteddata in AugmentedDataList:
                    with open(augmented_csv, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(Augmenteddata)
                        data_num += 1
        return data_num

    def visualize_graph(self, graph, node_labels, save_graph_name=None, show_graph=True):
        # plt.close()
        G = to_networkx(graph, node_attrs=['x'], edge_attrs=['edge_attr'])
        mapping = {k: v for k, v in zip(G.nodes, node_labels)}
        G = nx.relabel_nodes(G, mapping)
        c_list = ['skyblue' if n=='face' else 'orange' for n in G.nodes()]
        nx.draw_spring(G, with_labels=True, width = 3, edge_color="gray", node_color=c_list, node_size=2000)
        if save_graph_name is not None:
            plt.savefig(save_graph_name)
        if show_graph:
            # plt.show()
            plt.pause(0.1)
            plt.clf()

if __name__ == '__main__':
    user_name = input('enter user name')
    
    ft_path = os.path.dirname(__file__) +'/w2v_model/cc.en.300.bin'
    graph_utils = graph_utilitys(fasttext_model=ft_path)
    user_dir = os.path.dirname(os.path.abspath(__file__))+ "/experiment_data/"+user_name+"/position_data"

    csv_path_list = {0:user_dir+'/pattern_0.csv',1:user_dir+'/pattern_1.csv',2:user_dir+'/pattern_2.csv',3:user_dir+'/pattern_3.csv'}
    datasets,_ = graph_utils.csv2graphDataset(csv_path_list)
    print(datasets)
    print(len(datasets))
