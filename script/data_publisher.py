#!/usr/bin/python3
# -*- coding: utf-8 -*-
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import Float32MultiArray
import tf as TF
import os
import csv
from datetime import datetime
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import sys
from yolo_object_detector import GetYoloObjectInfo
import pickle
import socket
import pyautogui as pag
import time
import traceback
import json
import shutil

class JointStateSbscriber(object):

    def __init__(self):
        self.topick_name = "/hsrb/joint_states"
        self.joint_positions = None
        self._state_sub = rospy.Subscriber(self.topick_name, JointState, self._callback)
        # Wait until connection
        rospy.wait_for_message(self.topick_name, JointState, timeout=10.0)

    def _callback(self, data):
        self.joint_positions = data.position
    
    def get_head_position(self):
        return  self.joint_positions[9], self.joint_positions[10]


class TF_Publisher():
    def __init__(self, exe_type, tf_broadcaster):
        if exe_type == 'xtion':  # for Xtion
            topic_name = "/camera/depth_registered/points"
            self.reference_tf = '/camera_depth_frame'

        elif exe_type == 'hsr_sim': # for HSR simulator
            topic_name = "/hsrb/head_rgbd_sensor/depth_registered/points"
            self.reference_tf = 'head_rgbd_sensor_link'
            
        elif exe_type == 'hsr': # for HSR
            topic_name = '/hsrb/head_rgbd_sensor/depth_registered/rectified_points'
            self.reference_tf = 'head_rgbd_sensor_link'
        else:
            print('TF_Publisherクラスの初期化に失敗しました')
            sys.exit()
        self.exe_type = exe_type
        self.br = tf_broadcaster
        self.pc_sub = rospy.Subscriber(topic_name, PointCloud2, self.get_pc)
        rospy.wait_for_message(topic_name, PointCloud2, timeout=20.0)
        self.pc_data = None

    def get_pc(self, data):
        self.pc_data = data

    def create_object_TF(self, object_name, x, y, create=True):
        if self.pc_data is not None:
            pc_list = list(pc2.read_points(self.pc_data,skip_nans=True,
                                            field_names=('x', 'y', 'z'),
                                            uvs=[(x, y)]))
            if len(pc_list) != 0:
                x,y,z = pc_list[0]
                rot = [1.0, 0.0, 0.0, 0.0]
                if create and self.exe_type == 'xtion':
                    self.br.sendTransform([z,-x,-y], rot, rospy.Time.now(), '/'+object_name, self.reference_tf)
                if create and self.exe_type == 'hsr':
                    self.br.sendTransform([x,y,z], rot, rospy.Time.now(), '/'+object_name, self.reference_tf)
                return x,y,z
            else:
                return None, None, None
        else:
            return None, None, None
        

class MediapipePoseSubscriber():
    def __init__(self):
        self.pose = np.array([0.0]*32, dtype=float)
        self.face_pose = np.array([[0.0]*2]*11, dtype=float)
        self.face_pose_visibility = np.array([0.0]*11, dtype=float)
        self.face_sub = rospy.Subscriber("/mp_pose_data", Float32MultiArray, self.callback)
        rospy.wait_for_message("/mp_pose_data", Float32MultiArray, timeout=10.0)

    def callback(self, data):
        self.pose = np.reshape(data.data,(-1,3))

    def get_face_center(self):
        self.face_pose  = np.mean(self.pose[0:11,0:2], axis=0)
        self.face_pose_visibility = self.pose[0:11,2]
        return self.face_pose, self.face_pose_visibility

# ======================物体名と物体IDの設定======================
conf_dir = os.path.dirname(__file__)+'/obj_conf/'
MARKER_2_OBJECT ={}
OBJECT_NAME_2_ID ={}
ID_2_OBJECT_NAME = {}

# YOLOで認識可能な物体をさらにに限定する
obj_4_yolo = ["face", "tvmonitor", "laptop", "mouse", "keyboard", "book", "banana", "apple", "orange", "pizza","cup"]

for i, name in enumerate(obj_4_yolo):
    OBJECT_NAME_2_ID[name]=i
    ID_2_OBJECT_NAME[i] = name

# 設定をjsonファイルに書き込む
with open(conf_dir+'ID_2_OBJECT_NAME.json', 'w') as f:
    json.dump(ID_2_OBJECT_NAME, f)
with open(conf_dir+'OBJECT_NAME_2_ID.json', 'w') as f:
    json.dump(OBJECT_NAME_2_ID, f)

# ==================================================================

if __name__ == '__main__':
    rospy.init_node('data_publisher')
    
    user_name = rospy.get_param("/user_name")
    print('\n=============================')
    print('current user is '+user_name)
    print('=============================\n')
    rospy.sleep(3)
    user_dir = rospy.get_param("/user_dir")

    # 保存用ディレクトリの設定
    time_now = str(datetime.now()).split(' ')
    save_dir = user_dir  + '/'+time_now[0] + '-' +  time_now[1].split('.')[0].replace(':', '-')
    image_dir = save_dir+'/images/'
    position_dir = save_dir+'/position_data/'
    recognition_file_path = position_dir + '/data_when_recognition.csv'
    rospy.set_param("/save_dir", save_dir)
    rospy.set_param("/image_save_path", image_dir)

    try:
        os.makedirs(image_dir)
        os.makedirs(position_dir)
    except OSError:
        print('directory exist')

    # スクリーンショット保存用ディレクトリの作成（10パターン分）
    for i in range(10):
        try:
            os.makedirs(image_dir+'pattern_'+str(i))
        except OSError:
            print('directory exist')

    user_state_file = user_dir+"/state.csv"
    is_known_user = os.path.isfile(user_state_file)
    if not is_known_user:
        #状態管理用のファイル(csv)を新規作成
        with open(user_state_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['state'])
        print('created csv file for state database')
        try:
            os.makedirs(user_dir+'/PositionData_4_Train')
            os.makedirs(user_dir+'/learning_outputs')
        except OSError:
            print('directory exist')
    else:
        pass
    
    
    # データ送信のpublisher
    data_pub = rospy.Publisher("observed_data", Float32MultiArray, queue_size=1)

    # クラスのインスタンス化
    exe_type = rospy.get_param('exe_type')
    br = TF.TransformBroadcaster()
    listener = TF.TransformListener()  
    pose_sub = MediapipePoseSubscriber()
    yolo_info = GetYoloObjectInfo()
    tf_pub = TF_Publisher(exe_type=exe_type, tf_broadcaster=br)

    spin_rate=rospy.Rate(10)
    count_saved = 0
    pre_graph_data = None
    while not rospy.is_shutdown():

        robot_mode = rospy.get_param("/robot_mode")
        detectable_yolo_obj_lsit = obj_4_yolo

        obj_moved = False
        face_exist = False
        obj_positions =[]

        # ================================ グラフ用データの作成 ================================
        # MediaPipePoseの骨格検出による顔位置の取得
        face_center, visibility = pose_sub.get_face_center()
        if (visibility>0.5).all():
            try:
                face_x, face_y, face_z = tf_pub.create_object_TF('face', int(face_center[0]*640), int(face_center[1]*480), create=True)
                obj_positions.append([OBJECT_NAME_2_ID['face'], face_x, face_y, face_z])
                # print('face')
                face_exist = True
            except:
                traceback.print_exc()

        # 顔が検出できている時は物体ノードを作成する
        names = []
        if face_exist:
            # -------- object detection ---------
            objects_info = yolo_info.get_objects() #"/darknet_ros/detection_image"
            if len(objects_info) > 0:
                detect_obj_list = []
                for obj in objects_info:
                    name = obj[0]
                    if (name != 'person') and (name in detectable_yolo_obj_lsit):
                        x, y = obj[1], obj[2]
                        obj_x, obj_y, obj_z = tf_pub.create_object_TF(name, x, y, create=True)
                        if obj_x is not None:
                            obj_positions.append( [OBJECT_NAME_2_ID[name] ,obj_x, obj_y, obj_z] )
                            # print(name)
                            names.append(name)
                            pass
        else:
            pass

        graph_data = np.array(obj_positions).reshape(1,-1)[0].tolist()

        # Add data_id at the beginning of graph_data
        data_id = int(float(time.time())*100)
        graph_data.insert(0, float(data_id))
        # ====================================================================================

        # Check for changes in the position of faces and objects.
        try:
            graph_diff = np.array(pre_graph_data[1:]) - graph_data[1:]
            all_obj_moved = map(lambda k: abs(k)>0.02, graph_diff) # 前回保存したデータと比較して各オブジェクトが2cm以上移動しているかどうか
            if any(all_obj_moved):
                obj_moved = True
            else:
                obj_moved = False
        except (TypeError, ValueError):
            obj_moved = True
            pass            
        except:
            traceback.print_exc()


        #------------ グラフデータを送る ------------#
        if None in graph_data:
            continue
        msg_data = Float32MultiArray(data=graph_data)
        data_pub.publish(msg_data)
        #----------------------------------------------#
        node_num = (len(graph_data)-1)/4


        if robot_mode == 'graph_collecting':

            state_index = rospy.get_param("/state_index")
            state_name = rospy.get_param("/collecting_state_name")

            if face_exist and (node_num >= 2) and obj_moved:
                data_save_path = rospy.get_param("/data_save_path") # pattern_n.csv

                # save data
                with open(data_save_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(graph_data)
                pre_graph_data = graph_data
                count_saved += 1

                # Screenshot when data is saved
                image_save_path = rospy.get_param("/image_save_path")
                pag.screenshot(image_save_path+str(data_id)+'.jpg')
            else:
                pass
            print('number of  Data : '+str(count_saved))

            # Data collecting stops when the number of stored data reaches 1000.
            if count_saved >= 1000:
                save_dir = rospy.get_param("/save_dir")
                image_save_path = save_dir+'/images/'
                rospy.set_param("/image_save_path", image_save_path)
                rospy.set_param("/robot_mode", "finish_collecting")
                rospy.set_param("/cllecting_state_name", '')

        # 認識モードのとき
        elif robot_mode == 'state_recognition':
            with open(recognition_file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(graph_data)
                #　スクリーンショット
                # image_save_path = rospy.get_param("/image_save_path")
                # pag.screenshot(image_save_path+str(data_id)+'.jpg')
            print(names)
            # essential_obj_list = None

        # それ以外のモードのとき
        else:
            count_saved = 0
            # count_ideal_saved = 0
            # essential_obj_list = None
            
        spin_rate.sleep()
 