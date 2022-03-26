#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
 
import fasttext as ft
import fasttext.util

model_path = os.path.dirname(__file__)+'cc.en.300.bin'
if not os.path.exists(model_path):
    print("loading "+model_path)
    fasttext.util.download_model('en', if_exists='ignore')
