# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:59:54 2023

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
data_dict = {}
for name in sub_sheet_name:
    data_dict[name] = pd.read_csv('Data_A/' + name + '_EC.csv')
    data_dict[name]["time"] = pd.to_datetime(data_dict[name]["time"])
    
# import weather info

wh = pd.read_csv("Data_A/天气.csv")
wh["日期"] = pd.to_datetime(wh["日期"])
wh["time"] = wh["日期"] + pd.to_timedelta(wh["小时"],"h")
#wh.drop(["Unnamed:0","日期","小时"], axis=1, inplace=True)

# missing value processing

data = pd.merge(data_dict["二层插座"], data_dict["二层照明"], on = 'time', how = 'inner', suffixes=('_socket_2','_light_2'))
data = pd.merge(data,data_dict["一层插座"],on='time',how='inner',suffixes=('','_socket_1'))
data = pd.merge(data,data_dict["一层照明"],on='time',how='inner',suffixes=('','_light_1'))
data = pd.merge(data,data_dict["空调"],on='time',how='inner',suffixes=('','_air'))
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
'''
data.plot('time','value_socket',figsize=(18,5))
data.plot('time','value_light',figsize=(18,5))
data.plot('time','value_air',figsize=(18,5))
data.drop('time',axis=1,inplace=True)
'''

'''
预测插座消耗并计算分数

'''

# 多步预测 
data_socket = data.copy()
for i in range(7*24):
    data_socket['value_socket_{}'.format(i)] = data_socket['value_socket'].shift(-i-1)
data_socket.dropna(inplace=True)


#划分训练集，测试

targets = [item for item in data_socket.columns if 'value_socket_' in item]
targets_drop = ['Unnamed: 0','日期','time']

X_train_socket = data_socket.drop(targets,axis=1)[:int(len(data_socket)*0.95)]
y_train_socket = data_socket[targets][:int(len(data_socket)*0.95)]

X_test_socket = data_socket.drop(targets,axis=1)[int(len(data_socket)*0.95):]
y_test_socket = data_socket[targets][int(len(data_socket)*0.95):]
X_train_socket = X_train_socket.drop(targets_drop,axis=1)
X_test_socket = X_test_socket.drop(targets_drop,axis=1)



# 构建自回归模型
model_socket = MultiOutputRegressor(lgb.LGBMRegressor(objective='regression')).fit(X_train_socket,y_train_socket)

# 使用测试集测试
pred_socket = pd.DataFrame(model_socket.predict(X_test_socket),columns=targets)
pred_socket.index = y_test_socket.index



pred_num_socket = pred_socket.loc[15000,:].tolist()
y_test_num_socket = y_test_socket.loc[15000,:].tolist()

plt.figure(figsize=(18,5))
plt.plot(y_test_num_socket,label='test')
plt.plot(pred_num_socket,label='pred')
plt.legend()

# 分别计算七天的R2分数，大赛要求计算 R2_T_socket
R2_list_socket = []
weight_socket = [0.25,0.15,0.15,0.15,0.1,0.1,0.1]
for day in range(7):
    day_list = []
    for i in range(day*24,(day+1)*24):
        day_list.append('value_socket_{}'.format(i))
    pred_day_socket = pred_socket[day_list]
    test_day_socket = y_test_socket[day_list]
    R2_list_socket.append(r2_score(test_day_socket,pred_day_socket))

R2_list_socket = np.multiply(np.array(weight_socket),np.array(R2_list_socket)).sum()

'''
预测照明消耗并计算分数

'''

# 多步预测 
data_light = data.copy()
for i in range(7*24):
    data_light['value_light_{}'.format(i)] = data_light['value_light'].shift(-i-1)
data_light.dropna(inplace=True)


#划分训练集，测试
targets = [item for item in data_light.columns if 'value_light_' in item]

X_train_light = data_light.drop(targets,axis=1)[:int(len(data_light)*0.95)]
y_train_light = data_light[targets][:int(len(data_light)*0.95)]

X_test_light = data_light.drop(targets,axis=1)[int(len(data_light)*0.95):]
y_test_light = data_light[targets][int(len(data_light)*0.95):]


# 同插座相同处理
X_train_light = X_train_light.drop(targets_drop,axis=1)
X_test_light = X_test_light.drop(targets_drop,axis=1)


# 构建自回归模型
model_light = MultiOutputRegressor(lgb.LGBMRegressor(objective='regression')).fit(X_train_light,y_train_light)

# 使用测试集测试
pred_light = pd.DataFrame(model_light.predict(X_test_light),columns=targets)
pred_light.index = y_test_light.index

pred_num_light = pred_light.loc[15000,:].tolist()
y_test_num_light = y_test_light.loc[15000,:].tolist()


# 分别计算七天的R2分数，大赛要求计算 R2_T_light
R2_list_light = []
weight_light = [0.25,0.15,0.15,0.15,0.1,0.1,0.1]
for day in range(7): 
    day_list = []
    for i in range(day*24,(day+1)*24):
        day_list.append('value_light_{}'.format(i))
    pred_day_light = pred_light[day_list]
    test_day_light = y_test_light[day_list]
    R2_list_light.append(r2_score(test_day_light,pred_day_light))

R2_list_light = np.multiply(np.array(weight_light),np.array(R2_list_light)).sum()

'''
预测空调消耗并计算分数

'''

# 多步预测 
data_air = data.copy()
for i in range(7*24):
    data_air['value_air_{}'.format(i)] = data_air['value_air'].shift(-i-1)
data_air.dropna(inplace=True)


#划分训练集，测试
targets = [item for item in data_air.columns if 'value_air_' in item]

X_train_air = data_air.drop(targets,axis=1)[:int(len(data_air)*0.95)]
y_train_air = data_air[targets][:int(len(data_air)*0.95)]

X_test_air = data_air.drop(targets,axis=1)[int(len(data_air)*0.95):]
y_test_air = data_air[targets][int(len(data_air)*0.95):]

# 将多余数据去掉
X_train_air = X_train_air.drop(targets_drop,axis=1)
X_test_air = X_test_air.drop(targets_drop,axis=1)

# 构建自回归模型
model_air = MultiOutputRegressor(lgb.LGBMRegressor(objective='regression')).fit(X_train_air,y_train_air)

# 使用测试集测试
pred_air = pd.DataFrame(model_air.predict(X_test_air),columns=targets)
pred_air.index = y_test_air.index

# 分别计算七天的R2分数，大赛要求计算 R2_T_air
R2_list_air = []
weight_air = [0.25,0.15,0.15,0.15,0.1,0.1,0.1]
for day in range(7):
    day_list = []
    for i in range(day*24,(day+1)*24):
        day_list.append('value_air_{}'.format(i))
    pred_day_air = pred_air[day_list]
    test_day_air = y_test_air[day_list]
    R2_list_air.append(r2_score(test_day_air,pred_day_air))

R2_list_air = np.multiply(np.array(weight_air),np.array(R2_list_air)).sum()


# 计算总能耗，并计算分数

for i in range(7*24):
    y_test_air = y_test_air.rename(columns={'value_air_{}'.format(i):'value_total_{}'.format(i)})
    pred_air = pred_air.rename(columns={'value_air_{}'.format(i):'value_total_{}'.format(i)})

    y_test_light = y_test_light.rename(columns={'value_light_{}'.format(i):'value_total_{}'.format(i)})
    pred_light = pred_light.rename(columns={'value_light_{}'.format(i):'value_total_{}'.format(i)})

    y_test_socket = y_test_socket.rename(columns={'value_socket_{}'.format(i):'value_total_{}'.format(i)})
    pred_socket = pred_socket.rename(columns={'value_socket_{}'.format(i):'value_total_{}'.format(i)})

y_test_total = y_test_air + y_test_light + y_test_socket
pred_total = pred_socket + pred_light + pred_air

R2_list_total = []
weight_total = [0.25,0.15,0.15,0.15,0.1,0.1,0.1]
for day in range(7):
    day_list = []
    for i in range(day*24,(day+1)*24):
        day_list.append('value_total_{}'.format(i))
    pred_day_total = pred_total[day_list]
    test_day_total = y_test_total[day_list]
    R2_list_total.append(r2_score(test_day_total,pred_day_total))

R2_list_total = np.multiply(np.array(weight_total),np.array(R2_list_total)).sum()

# 预测未来七天的各项能耗，并输出用于评测的 csv文件

final_test_input = data.iloc[-1:]
final_test_input = final_test_input.drop(targets_drop,axis=1)

final_pred_socket = model_socket.predict(final_test_input)
final_pred_light = model_light.predict(final_test_input)
final_pred_air = model_air.predict(final_test_input)

results = pd.DataFrame()
results['total'] = final_pred_air[0] + final_pred_light[0] + final_pred_socket[0]
results['air'] = final_pred_air[0]
results['light'] = final_pred_light[0]
results['socket'] = final_pred_socket[0]

results.to_csv('prediction_results.csv',index=False)


