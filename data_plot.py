# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:32:13 2023

@author: gaoji
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import numpy as np
import datetime 
from chinese_calendar import is_workday
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# import energy consumption data
sub_sheet_name = ["二层插座","二层照明","一层插座","一层照明","空调"]
datadict = {}
for name in sub_sheet_name:
    datadict[name] = pd.read_csv('dataA/' + name + '_EC.csv')
    datadict[name]["time"] = pd.to_datetime(datadict[name]["time"])
    
# import weather info

wh = pd.read_csv("dataA/天气.csv")
wh["日期"] = pd.to_datetime(wh["日期"])
wh["time"] = wh["日期"] + pd.to_timedelta(wh["小时"],"h")
#wh.drop(["Unnamed:0","日期","小时"], axis=1, inplace=True)

# missing value processing

data = pd.merge(datadict["二层插座"], datadict["二层照明"], on = 'time', how = 'inner', suffixes=('_socket_2','_light_2'))
data = pd.merge(data,datadict["一层插座"],on='time',how='inner',suffixes=('','_socket_1'))
data = pd.merge(data,datadict["一层照明"],on='time',how='inner',suffixes=('','_light_1'))
data = pd.merge(data,datadict["空调"],on='time',how='inner',suffixes=('','_air'))
data = data.rename(columns={"value":"value_socket_1"})


#一层二层插座能耗相加，一层二层照明能耗相加。构造数据
data["value_socket"] = data["value_socket_2"] + data["value_socket_1"]
data["value_light"] = data["value_light_2"] + data["value_light_1"]
data.drop(['value_socket_2','value_light_2','value_socket_1','value_light_1'],axis=1,inplace=True)


# 数据尺度调整为每小时，根据赛题要求选定数据范围，构造特征

data = data[(data['time'] >= '2013-8-03 00:00:00') &( data['time'] <= '2015-08-03 00:00:00')]
if_not_workday = []
for dat in data['time']:
    if_not_workday.append(is_workday(dat))
data['workday'] = if_not_workday
data['hour'] = data['time'].dt.hour
data['week'] = data['time'].dt.weekday
data['day']  = data['time'].dt.day
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year
data = data.groupby(['year','month','day','week','hour','workday'])[['value_socket','value_light','value_air']].sum().reset_index()


# 处理天气数据，进行merge
wh['hour'] = wh['time'].dt.hour
wh['week'] = wh['time'].dt.weekday
wh['day']  = wh['time'].dt.day
wh['month'] = wh['time'].dt.month
wh['year'] = wh['time'].dt.year
data = pd.merge(data,wh,on=['hour','week','day','month','year'],how='inner')
data = data.rename(columns={"温度":'temp',"湿度":'humidity',"降雨量":'rainfall',
                            "大气压":'atmos',"风向":'wind_direction',
                            "风向角度":'wind_angle',"风速":'wind_speed',"云量":'cloud'})
le = LabelEncoder()
data['wind_direction'] = le.fit_transform(data['wind_direction'])
data['wind_direction'] = data['wind_direction'].astype('category')
data['workday'] = le.fit_transform(data['workday'])
data['work'] = data['workday'].astype('category')

# 绘制图形

data.plot('time','value_socket',figsize=(18,5))
data.plot('time','value_light',figsize=(18,5))
data.plot('time','value_air',figsize=(18,5))
data.drop('time',axis=1,inplace=True)


data[-168:].to_csv('air.csv',columns=(['value_air','value_socket','value_light']))