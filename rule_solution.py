# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.models import *

def plot_wifi_records(WIFI_Records):
    print("plot_wifi_records begain")
    # WIFI_everyArea = WIFI_Records.groupby(by=["WIFIAPTag"])["passengerCount"].apply().reset_index()
    WIFI_everyArea = WIFI_Records.sort_values(by=["WIFIAPTag", "timeStamp"])
    WIFI_everyArea.to_csv("clean_data/WIFI_area_sort.csv", index=False)
    groups = WIFI_Records.groupby("WIFIAPTag")
    for num, APTag in groups:
        # 画图部分
        sns.set(rc={"figure.figsize": (12, 8)})  # 图像的大小
        sns.set_context(font_scale=3, rc={"lines.linewidth": 2.5})
        APTag.plot(x="timeStamp", y="passengerCount")
        plt.title(str(groups[num][0]))  # 图的标题
        plt.xlabel("time")  # 横坐标
        plt.ylabel("passengerCount")  # 纵坐标
        plt.savefig("figure/" + str(groups[num][0]) + ".jpg", facecolor='w')
    plt.close()
    print("plot_wifi_records Done")


def rule_solution():
    print("Run in rule_solution!")
    '''
    WIFI_Records = pd.read_csv("../input/WIFI_AP_Passenger_Records_chusai_1stround.csv",
                               parse_dates=['timeStamp'],
                               date_parser=(lambda dt: pd.Timestamp(dt))
                               )
    '''
    WIFI_Records = pd.read_csv("../input/WIFI_AP_Passenger_Records_chusai_1stround.csv")
    # WIFI_Records["timeStamp"] = WIFI_Records["timeStamp"].apply(lambda x: pd.Timestamp(x))
    pre_time = pd.date_range(start="20160914150000", end="20160914175959", freq="10min")
    pre_WIFIAPTag = WIFI_Records["WIFIAPTag"].drop_duplicates()
    plot_wifi_records(WIFI_Records)  # 每个wifi接入点中，时间与人数的关系

    # 方法一：取均值
    rule = RULES(WIFI_Records, pre_time, pre_WIFIAPTag)
    rule_res_mean = rule.mean_solution()
    return rule_res_mean

if __name__ == "__main__":

    rule_result = rule_solution()
    rule_result.to_csv("submissions/airport_gz_passenger_predict_mean.csv", index=False)