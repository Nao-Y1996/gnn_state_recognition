#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import socket
import pickle
from graph_tools import graph_utilitys
from classificator_nnconv import classificator
import traceback
import json


graph_utils = graph_utilitys(fasttext_model=os.path.dirname(os.path.abspath(__file__)) +'/w2v_model/cc.en.300.bin')
detectable_obj_num = len(graph_utils.ID_2_OBJECT_NAME.keys())
# all_obj_names = graph_utils.ID_2_OBJECT_NAME.values()


class DataSubscriber():
    def __init__(self):
        self.observed_data = [0.0, 0.0, 0.0, 0.0]
        self.data_sub = rospy.Subscriber("/observed_data", Float32MultiArray, self.callback)
        # rospy.wait_for_message("/observed_data", Float32MultiArray, timeout=10.0)

    def callback(self, data):
        self.observed_data = data.data

    def get_data(self):
        return self.observed_data


if __name__ == '__main__':

    rospy.init_node('model_nnconv', anonymous=True)
    spin_rate=rospy.Rate(20)

    user_name = rospy.get_param("/user_name")

    # -------------------------- Initialization of recognition model --------------------------
    model_path = '/home/'+ os.getlogin() +'/catkin_ws/src/gnn_state_recognition/script/recognition_model/'+user_name+'_model.pt'
    model_info_path = '/home/'+ os.getlogin() +'/catkin_ws/src/gnn_state_recognition/script/recognition_model/'+user_name+'_model_info.json'
    model_update_time, modelinfo_update_time = None, None
    model_exist = os.path.exists(model_path) and os.path.exists(model_info_path)
    if model_exist:
        model_update_time = os.path.getmtime(model_path)
        modelinfo_update_time = os.path.getmtime(model_info_path)
        print('The Recognition model settings loaded.')

        with open(model_info_path) as f:
            _dict = json.load(f)
            pattern_num = _dict['pattern_num']
        cf = classificator(model=model_path, output_dim=pattern_num)
        
        data_buf_len = 10
        count4probability = 0
        probability_list = np.array([[0.0]*pattern_num] * data_buf_len)

        model_loaded = True
    else:
        print('No recognition model exists yet.')
        model_loaded = False
    # -------------------------------------------------------------------- 

    data_sub = DataSubscriber()
    probability_pub = rospy.Publisher("avarage_probability", Float32MultiArray, queue_size=1)

    while not rospy.is_shutdown():
        robot_mode = rospy.get_param("/robot_mode")
        model_exist = os.path.exists(model_path) and os.path.exists(model_info_path)
        if model_exist:
            # --------------------- When the model is updated, read the model settings and initialize various variables ---------------------
            if model_update_time != os.path.getmtime(model_path) and modelinfo_update_time != os.path.getmtime(model_info_path):
                model_update_time = os.path.getmtime(model_path)
                modelinfo_update_time = os.path.getmtime(model_info_path)
                print('The recognition model has been updated.')

                with open(model_info_path) as f:
                    _dict = json.load(f)
                    pattern_num = _dict['pattern_num']
                    model_name = _dict['model_name']
                cf = classificator(model=model_path, output_dim=pattern_num)
                
                data_buf_len = 10
                count4probability = 0
                probability_list = np.array([[0.0]*pattern_num] * data_buf_len)

                model_loaded = True
            else:
                model_loaded = True
        else:
            model_loaded = False
        # ---------------------------------------------------------------------------------------------------
        if model_loaded:
            if robot_mode == 'state_recognition':
                
                # subscribe data
                data = data_sub.get_data()
                
                # convert data to graph
                position_data = graph_utils.removeDataId(data)
                graph, node_names = graph_utils.positionData2graph(position_data, 10000, include_names=True)
                if graph is not None:

                    # state recognition
                    probability = cf.classificate(graph)

                    # Calculate the average of recognition probabilities（過去data_buf_len個分のデータで平均を取る）
                    probability_list[count4probability] = probability
                    average_probability  = probability_list.mean(axis=0).tolist()
                    count4probability += 1
                    if count4probability >= data_buf_len:
                        count4probability = 0

                else: # graph is None
                    average_probability = [0.0] * pattern_num
                    pass
            
            else:
                average_probability = [0.0] * pattern_num
                pass

            #　Publish recognition results (probability)
            msg_average_probability = Float32MultiArray(data=average_probability)
            probability_pub.publish(msg_average_probability)

        else:
            pass

        spin_rate.sleep()