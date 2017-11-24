# -*- coding:utf-8 -*-
import sys
import os

if getattr(sys, 'frozen', False):
    ROOT_PATH = os.path.dirname(os.path.dirname(sys.executable))
else:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_PATH = os.path.join(ROOT_PATH, 'log')
RECORD_PATH = os.path.join(ROOT_PATH, 'record')

for path in [LOG_PATH, RECORD_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)
