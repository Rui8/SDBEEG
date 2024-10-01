# -- coding: utf-8 --
# @Time : 2021/11/30 20:48
# @Author : Shurui
# @File : myaml.py
# @Software: PyCharm
import yaml
from dotmap import DotMap

def load_config(config_file_path):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    return DotMap(config)