#!/usr/bin/python3
# -*- coding: utf-8 -*-
import socket
import sys
import rospy
import os
import pandas as pd
import csv
import time
from robot_tools import RobotPartner


def exist_state_check(csv_file, state_name):
    df = pd.read_csv(csv_file)
    state_indexs = list(df[df['state']==state_name].index)
    if len(state_indexs) == 0:
        return False
    elif len(state_indexs) == 1:
        return True
    else:
        return None

def get_state_index(csv_file, state_name):
    df = pd.read_csv(csv_file)
    state_indexs = list(df[df['state']==state_name].index)
    if len(state_indexs) == 0:
        state_index = len(df)
    elif len(state_indexs) == 1:
        state_index = state_indexs[0]
    else:
        state_index = None
    return state_index

def get_stateName(csv_file, state_id):
    df = pd.read_csv(csv_file)
    name = df.iat[state_id, 0]
    return name

def add_new_state(csv_file, state_name):
    exist_state = exist_state_check(csv_file, state_name)
    if not exist_state:
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([state_name])
        print (f'------ A new state added : {state_name} ------')
    else:
        pass


if __name__ == "__main__":

    # 音声認識結果を受け取るための通信の設定
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP_ADDRESS = s.getsockname()[0] # get IP address of this PC
    port = 8880
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
    sock.bind((IP_ADDRESS, port))
    rospy.set_param('ip_address', str(IP_ADDRESS))
    rospy.set_param('port', port)

    """
    実行タイプ
    hsr     : real robot
    hsr_sim : HSR simulator
    xtion   : xtion camera
    """
    exe_type = rospy.get_param('exe_type')
    
   
    if exe_type=='hsr' or exe_type=='hsr_sim':
        # HSRロボット機能のimport
        from hsrb_interface import Robot
        hsr_robot = Robot()
        robotPartner = RobotPartner(exe_type=exe_type,hsr_robot=hsr_robot)
    elif exe_type=='xtion':
        robotPartner = RobotPartner(exe_type=exe_type,hsr_robot=None)
    else:
        pass

    # usernameの入力
    user_name = input('\nenter user name\n')
    
    rospy.set_param("/user_name", user_name)
    rospy.set_param("/IP_address", str(IP_ADDRESS))
    rospy.set_param("/robot_mode", "nomal")

    # 保存用ディレクトリの設定
    user_dir = os.path.dirname(__file__)+'/experiment_data/'+user_name
    rospy.set_param("/user_dir", user_dir)

    print('-------- start --------\n')
    while not rospy.is_shutdown():
        state_name = None
        state_index = None
        robot_mode = rospy.get_param("/robot_mode")
        try :
            
            message, cli_addr = sock.recvfrom(1024)
            
            message = message.decode(encoding='utf-8')
            print(message)

            if ('はい' in message) or ('します'in message) or ('現在'in message) or ('今'in message) or ('完了'in message):
                pass
            
            elif 'モードの確認' == message or ("check the mode" in message):
                robot_mode = rospy.get_param("/robot_mode")
                if robot_mode == "state_recognition":
                    robotPartner.say('現在、認識モードです。')
                    # robotPartner.say("It's recognition mode.")
                elif robot_mode == "nomal":
                    robotPartner.say('現在、通常モードです。')
                    # robotPartner.say("It's nomal mode. ")
                elif robot_mode == "waite_state_name":
                    robotPartner.say('現在、記録の準備中です。 今、何をしているか教えてもらえたら記録を開始できます。')
                    # robotPartner.say("Currently, I am preparing to record data. Please let me know what you are doing now so I can start recording.")
                elif robot_mode == "graph_collecting":
                    robotPartner.say('現在、データを記録中です。')
                    # robotPartner.say('Now I am recording data.')
                elif robot_mode == "auto_train":
                    robotPartner.say('現在、モデルの学習中です。学習が完了するまでお待ちください')
                    # robotPartner.say('The model is currently being trained. Please wait until learning is complete.')
                else:
                    robotPartner.say('モードが不明です。プログラムを修正する必要があるかもしれません。')
                    # robotPartner.say('Sorry, the mode is unknown. You may need to modify the program.')

            elif '終了' == message or ("finish recording" in message):
                if (robot_mode=='graph_collecting'):
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = rospy.get_param("/state_index")
                    state_name = get_stateName(db_file, state_index)
                    
                    image_save_path = save_dir+'/images/'
                    rospy.set_param("/image_save_path", image_save_path)
                    rospy.set_param("/robot_mode", "nomal")
                    rospy.set_param("/cllecting_state_name", '')

                    robotPartner.say(state_name + 'の記録は完了です。')
                    # robotPartner.say('Recording your state data of ' + state_name + 'is finished.')
                else:
                    pass
            
            elif '認識モード'== message or ('start recognition' in message):
                can_recognize = os.path.exists(user_dir+'/model_info.json')
                if can_recognize:
                    rospy.set_param("/robot_mode", "state_recognition")
                    robotPartner.say('はい、認識機能をオンにします。')
                    # robotPartner.say('Yes, the recognition function is turned on.')
                else:
                    robotPartner.say('利用可能な認識モデルがありません。学習後に認識モードが利用可能になります。')
                    # robotPartner.say('There are no recognition models available. Recognition mode will be available after training.')
            
            elif 'モデルの学習を開始'==message or ('start training the model' in message):
                robotPartner.say('モデルの学習を開始します')
                # robotPartner.say('Yes, start training the model.')
                time.sleep(1)

                rospy.set_param("/robot_mode", "auto_train")
            
            elif '通常モード' == message or ('back to normal mode' in message):
                rospy.set_param("/robot_mode", "nomal")
                robotPartner.say('はい、通常機能に戻ります。')
                # robotPartner.say('Yes, back to normal mode.')

            elif '記録して' == message or ('start recording' in message):
                rospy.set_param("/robot_mode", "waite_state_name")
                robotPartner.say('はい、今何をしていますか？')
                # robotPartner.say('Yes, what are you doing now?')

            elif 'プログラム終了' == message:
                break
                # sys.exit('プログラム終了')
            else :
                robot_mode = rospy.get_param("/robot_mode")
                if (robot_mode=='waite_state_name'):
                    state_name = message
                    robotPartner.say(state_name + '、を記録します。')
                    # robotPartner.say('Recording state data of '+state_name )
                    
                    save_dir = rospy.get_param("/save_dir")
                    user_dir = rospy.get_param("/user_dir")
                    db_file = user_dir+"/state.csv"

                    state_index = get_state_index(db_file, state_name)
                    add_new_state(db_file, state_name)

                    rospy.set_param("/state_index", state_index)

                    # 収集するデータを保存するファイルを指定
                    image_save_path = save_dir+'/images/pattern_'+str(state_index)+'/'
                    data_save_path = save_dir+'/position_data/pattern_'+str(state_index)+'.csv'
                    rospy.set_param("/data_save_path", data_save_path)
                    rospy.set_param("/image_save_path", image_save_path)
                    
                    # データ収集モードに切り替え
                    rospy.set_param("/robot_mode", "graph_collecting")
                    rospy.set_param("/collecting_state_name", state_name)
                    print(f'------ start collecting data of "{state_name}" ------')
                elif (robot_mode=='finish_collecting'):
                    robotPartner.say('記録は完了です')
                    rospy.set_param("/robot_mode", "nomal")
                else:
                    pass



        except KeyboardInterrupt:
            print ('\n . . .\n')
            if sock is not None:
                sock.close()
            break
        except socket.timeout:
            pass
        except:
            import traceback
            traceback.print_exc()
        
