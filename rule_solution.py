# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
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


def rule_solution(filename, start_time, end_time):
    print("Run in rule_solution!")
    '''
    WIFI_Records = pd.read_csv("../input/WIFI_AP_Passenger_Records_chusai_1stround.csv",
                               parse_dates=['timeStamp'],
                               date_parser=(lambda dt: pd.Timestamp(dt))
                               )
    '''
    WIFI_Records = pd.read_csv(filename)
    # WIFI_Records["timeStamp"] = WIFI_Records["timeStamp"].apply(lambda x: pd.Timestamp(x))
    pre_time = pd.date_range(start=start_time, end=end_time, freq="10min")
    pre_WIFIAPTag = WIFI_Records["WIFIAPTag"].drop_duplicates()
    #  plot_wifi_records(WIFI_Records)  # 每个wifi接入点中，时间与人数的关系
    rule = RULES(WIFI_Records, pre_time, pre_WIFIAPTag)

    # 方法一：取均值一
    # rule_res_mean = rule.mean_solution_one()
    # 方法二：取均值二
    rule_res_mean = rule.mean_solution_two()

    return rule_res_mean

def test(result):
    test_result = pd.read_csv("clean_data/airport_gz_passenger_predict_standard.csv")
    Criteria = sum((result['passengerCount']-test_result['passengerCount'])*(result['passengerCount']-test_result['passengerCount']))
    print("Criteria = %f" % Criteria)

if __name__ == "__main__":
    time0 = time.time()
    '''
    # 测试集
    filename = "clean_data/test_train.csv"
    start_time = "20160914120000"
    end_time = "20160914145959"
    '''


    # 真正训练集
    filename = "../input/WIFI_AP_Passenger_Records_chusai_1stround.csv"
    start_time = "20160914150000"
    end_time = "20160914175959"


    rule_result = rule_solution(filename, start_time, end_time)
    rule_result.to_csv("submissions/airport_gz_passenger_predict_mean_two.csv", index=False)

    # test(rule_result)
    time1 = time.time()
    print ("It take %f s to process" % (time1 - time0))