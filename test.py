# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from common.models import *
from pandas.tseries.offsets import Hour, Minute
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from dateutil.parser import parse

train = pd.read_csv("../input/WIFI_AP_Passenger_Records_chusai_2ndround.csv")
train["slice10min"] = train["timeStamp"].apply(lambda x: x[0:15])
train = train.groupby(by=["WIFIAPTag", "slice10min"])["passengerCount"].sum().reset_index()
train["passengerCount"] = (train["passengerCount"] / 10).round(2)
train.to_csv("clean_data/train.csv", index=False)
# 横向画图，每个ｗｉｆｉ点在不同时间，相同时刻的图像
train["time"] = train["slice10min"].apply(lambda x: x[-4:])
WIFIs = train.groupby(by=["WIFIAPTag", "time"])
i = 1
for (name, time), WIFI in WIFIs:
    print ("开始画图%d" % i)
    i = i+1
    sns.set(rc={"figure.figsize": (12, 8)})  # 图像的大小
    sns.set_context(font_scale=3, rc={"lines.linewidth": 2.5})
    WIFI.plot(x="slice10min", y="passengerCount")
    plt.title(str(name)+"-"+ str(time))  # 图的标题
    plt.xlabel("day")  # 横坐标
    plt.ylabel("passengerCount")  # 纵坐标
    plt.savefig("figure/不同日期相同时刻/" + str(name)+"-"+ str(time) + ".jpg", facecolor='w')
plt.close()
print("横向画图 Done")