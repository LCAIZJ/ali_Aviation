# coding: utf-8
import numpy as np
import pandas as pd
import xgboost as xgb


class RULES(object):
    def __init__(self, WIFI_Records, pre_time, pre_WIFIAPTag):
        self.WIFI_Records = WIFI_Records
        self.pre_time = [w.strftime("%Y-%m-%d-%H-%M") for w in pre_time]
        for i in range(len(self.pre_time)):
            self.pre_time[i] = self.pre_time[i][0:-1]
        self.pre_WIFIAPTag = pre_WIFIAPTag
        print("RULES is inited OK!")

    def mean_solution(self):
        print("Mean_solution Run")
        pre_value = self.WIFI_Records.groupby("WIFIAPTag")["passengerCount"].mean().round(1).reset_index()
        # 转换成标准格式
        lst = list(1 for i in range(1, len(self.pre_time)+1))
        trans = pd.DataFrame({'slice10min':self.pre_time, 'flag':lst})
        pre_value['flag'] = 1
        pre_value = pre_value.merge(trans, how='left', on='flag')
        del pre_value['flag']
        pd.DataFrame(pre_value, columns=['PassengerCount', 'WIFIAPTag', 'timeStamp'])
        # pre_value.to_csv("submissions/airport_gz_passenger_predict_mean.csv", index=True)
        return pre_value




