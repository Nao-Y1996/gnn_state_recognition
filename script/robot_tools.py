#!/usr/bin/python
# -*- coding: utf-8 -*-
class RobotPartner():
    def __init__(self, exe_type, hsr_robot):
        self.exe_type = exe_type
        self.tts = None
        self.whole_body = None
        if self.exe_type == 'hsr':
            self.whole_body = hsr_robot.try_get('whole_body')
            self.tts = hsr_robot.try_get('default_tts')
            
    def say(self, content):
        if self.exe_type=='hsr':
            self.tts.say(content)
        else:
            print('Robot : ' + content)