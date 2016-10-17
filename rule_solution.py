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
    WIFI_Records = pd.read_csv(filename)

    # 构造一个测试集，9.25号12点到14点59
    '''
    test_time = ["2016-09-25-12", "2016-09-25-13", "2016-09-25-14"]
    WIFI_Records_test = WIFI_Records.loc[~WIFI_Records["timeStamp"].apply(lambda x: x[0:13]).isin(test_time),
                                   ["WIFIAPTag", "passengerCount", "timeStamp"]]
    WIFI_Records_test_result = WIFI_Records.loc[WIFI_Records["timeStamp"].apply(lambda x: x[0:13]).isin(test_time),
                                         ["WIFIAPTag", "passengerCount", "timeStamp"]]
    WIFI_Records_test_result["slice10min"] = WIFI_Records_test_result["timeStamp"].apply(lambda x: x[0:15])
    WIFI_Records_test_result = WIFI_Records_test_result.groupby(by=["WIFIAPTag", "slice10min"])["passengerCount"].sum().reset_index()
    WIFI_Records_test_result["passengerCount"] = (WIFI_Records_test_result["passengerCount"] / 10).round(2)
    WIFI_Records_test_result = pd.DataFrame(WIFI_Records_test_result, columns=['passengerCount', 'WIFIAPTag', 'slice10min'])
    WIFI_Records_test.to_csv("clean_data/test12-15.csv", index=False)
    WIFI_Records_test_result.to_csv("clean_data/test12-15_result.csv", index=False)
    '''
    pre_time = pd.date_range(start=start_time, end=end_time, freq="10min")
    pre_WIFIAPTag = WIFI_Records["WIFIAPTag"].drop_duplicates()
    #  plot_wifi_records(WIFI_Records)  # 每个wifi接入点中，时间与人数的关系
    rule = RULES(WIFI_Records, pre_time, pre_WIFIAPTag)
    # 方法一：取均值一
    # rule_res = rule.mean_solution_one()
    # 方法二：取均值二
    rule_res = rule.mean_solution_two()
    # 方法三：取中值
    # rule_res = rule.median_solution()
    return rule_res

def extract_features(train_df, test_df):
    # Combine train and test set
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    df = pd.concat([train_df, test_df], ignore_index=True)
    # df.to_csv("clean_data/df.csv", index=False)
    # 去除11号的数据
    # df = df.loc[~df["slice10min"].apply(lambda x: x[8:10]).isin(["11"]), ["WIFIAPTag","slice10min","passengerCount","is_train"]]
    features_x = ["WIFIAPTag"]
    features_y = "log_passengerCount"

    # log_y
    df["log_passengerCount"] = np.log1p(df["passengerCount"])

    # -----------------------------
    #  Add the time features
    # -----------------------------
    features_x.append('slice10min')

    # ------------------------------
    #  Add the bgate number features
    # ------------------------------
    bgate_df_org = pd.read_csv("../input/airport_gz_gates.csv")
    bgate_df = bgate_df_org.groupby(by=["BGATE_AREA"])["BGATE_ID"].count().reset_index()
    # print bgate_df
    bgate_df.rename(columns={"BGATE_ID":"BGATE_num"}, inplace = True)
    df["BGATE_AREA"] = df["WIFIAPTag"].apply(lambda x: x[0:2])
    df = df.merge(bgate_df, how='left', on=["BGATE_AREA"]).fillna(0)
    features_x.append("BGATE_num")

    # ------------------------------
    #  Add the mean features
    # ------------------------------

    # 使用前几天的均值
    # mean_df = train_df.loc[train_df["slice10min"].apply(lambda x: x[0:10]).isin(["2016-09-12, 2016-09-13, 2016-09-14"]),
    #                       ["passengerCount", "WIFIAPTag", "slice10min"]]
    '''
    train_df["time"] = train_df["slice10min"].apply(lambda x: x[-4:])

    mean_11 = train_df.loc[train_df["slice10min"].apply(lambda x: x[0:10]).isin(["2016-09-11"]),
                           ["passengerCount", "WIFIAPTag", "slice10min"]]
    mean_12 = mean_11
    mean_11.rename(columns={"passengerCount": "mean"}, inplace=True)
    mean_12["slice10min"] = mean_12["slice10min"].apply(lambda x: x[0:8]+'12'+x[10:])
    mean_12.rename(columns={"passengerCount":"mean"}, inplace=True)

    mean_13 = train_df.loc[train_df["slice10min"].apply(lambda x: x[0:10]).isin(["2016-09-11", "2016-09-12"]),
                           ["passengerCount", "WIFIAPTag", "slice10min", 'time']]
    mean_13 = mean_13.groupby(by=['time', 'WIFIAPTag'])["passengerCount"].mean().reset_index()
    mean_13["slice10min"] = mean_13["time"].apply(lambda x: "2016-09-13-"+x)
    mean_13.rename(columns={"passengerCount": "mean"}, inplace=True)
    del mean_13["time"]
    mean_14 = train_df.loc[train_df["slice10min"].apply(lambda x: x[0:10]).isin(["2016-09-11", "2016-09-12", "2016-09-13"]),
                           ["passengerCount", "WIFIAPTag", "slice10min", 'time']]
    mean_14 = mean_14.groupby(by=['time', 'WIFIAPTag'])["passengerCount"].mean().reset_index()

    mean_14["slice10min"] = mean_14["time"].apply(lambda x: "2016-09-14-" + x)
    mean_14.rename(columns={"passengerCount": "mean"}, inplace=True)
    # mean_14.to_csv("clean_data/mean_14.csv", index=False)
    del mean_14["time"]
    mean_df = pd.concat([mean_12, mean_13, mean_14], ignore_index=True)
    # mean_df.to_csv("clean_data/mean_df.csv", index=False)
    df = df.merge(mean_df, how='left', on=["WIFIAPTag", "slice10min"]).fillna(0)

    features_x.append("mean")

    # 使用了整体的均值
    '''
    train_df["time"] = train_df["slice10min"].apply(lambda x: x[-4:])
    mean_df = train_df.groupby(by=["time", "WIFIAPTag"])["passengerCount"].mean().reset_index()
    mean_df.rename(columns={"passengerCount":"mean"}, inplace=True)
    mean_df.to_csv("clean_data/mean_df.csv", index=False)
    df["time"] = df["slice10min"].apply(lambda x: x[-4:])
    df = df.merge(mean_df, how="left", on=["WIFIAPTag", "time"])
    features_x.append("mean")


    '''
    # 错误使用了14号的均值
    mean_df = pd.read_csv("clean_data/train_data.csv")
    mean_df.rename(columns={"passengerCount":"mean"}, inplace=True)
    del mean_df["slice10min"]
    df["time"] = df["slice10min"].apply(lambda x: x[-4:])
    df = df.merge(mean_df, how="left", on=["WIFIAPTag", "time"])
    del df["time"]
    features_x.append("mean")
    '''
    # -------------------------------
    #  Add the flights number features
    # -------------------------------

    flights_df_org = pd.read_csv("../input/airport_gz_flights_chusai_1stround.csv",
                             parse_dates=['scheduled_flt_time', 'actual_flt_time'],
                             date_parser=(lambda dt: pd.to_datetime(dt, format='%Y/%m/%d %H:%M:%S') + 8*Hour()))
    flights_df = flights_df_org
    flights_df_org.to_csv("clean_data/flights_df_org.csv", index=False)
    # flights_df["scheduled_flt_time"] = flights_df["scheduled_flt_time"].apply(lambda x: pd.Timestamp(x) + 8 * Hour())
    # flights_df["actual_flt_time"] = flights_df["actual_flt_time"].apply(lambda x: pd.Timestamp(x) + 8 * Hour())
    # print flights_df["scheduled_flt_time"]
    # 分情况讨论，如果有临时改变了登机口的，就按照最后一个登机口算
    '''
    flights_df["BGATE_ID"] = flights_df["BGATE_ID"].fillna(0)
    for i in range(len(flights_df["BGATE_ID"])):
        # print flights_df["BGATE_ID"][i]
        if flights_df["BGATE_ID"][i]!=0 and "," in flights_df["BGATE_ID"][i]:
                flights_df["BGATE_ID"][i] = flights_df["BGATE_ID"][i].split(",")[1]
    '''
    flights_df = flights_df.merge(bgate_df_org, how='left', on=["BGATE_ID"]).fillna(0)
    # flights_df[]
    # flights_df["scheduled_flt_time-30"] = flights_df["scheduled_flt_time"]- 0.5*Hour()
    # flights_df["actual_flt_time-20"] = flights_df["actual_flt_time"]-Hour()
    flights_df["beforeScheduled_60"] = flights_df["scheduled_flt_time"] - Hour()
    flights_df["beforeScheduled_60"] = flights_df["beforeScheduled_60"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    flights_df["beforeScheduled_50"] = flights_df["scheduled_flt_time"] - 50*Minute()
    flights_df["beforeScheduled_50"] = flights_df["beforeScheduled_50"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    flights_df["beforeScheduled_40"] = flights_df["scheduled_flt_time"] - 40*Minute()
    flights_df["beforeScheduled_40"] = flights_df["beforeScheduled_40"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    flights_df["beforeScheduled_30"] = flights_df["scheduled_flt_time"] - 30*Minute()
    flights_df["beforeScheduled_30"] = flights_df["beforeScheduled_30"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    flights_df["beforeScheduled_20"] = flights_df["scheduled_flt_time"] - 20*Minute()
    flights_df["beforeScheduled_20"] = flights_df["beforeScheduled_20"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    flights_df["beforeScheduled_10"] = flights_df["scheduled_flt_time"] - 10*Minute()
    flights_df["beforeScheduled_10"] = flights_df["beforeScheduled_10"].apply(
        lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # flights_df.to_csv("clean_data/flights_df.csv", index=False)

    flights_df_60 = flights_df.groupby(by=["beforeScheduled_60", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_60.rename(columns={'flight_ID':"flightNum_60", 'beforeScheduled_60':'slice10min'}, inplace = True)
    # flights_df_60["slice10min"] = flights_df_60["beforeScheduled_60"].apply(
    #    lambda x: str(x)[0:10]+"-"+str(x)[11:13]+"-"+str(x)[14:15])
    # del flights_df_60["beforeScheduled_60"]
    df = df.merge(flights_df_60, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)

    flights_df_50 = flights_df.groupby(by=["beforeScheduled_50", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_50.rename(columns={'flight_ID': "flightNum_50", 'beforeScheduled_50':'slice10min'}, inplace=True)
    # flights_df_50["slice10min"] = flights_df_50["beforeScheduled_50"].apply(
    #    lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # del flights_df_50["beforeScheduled_50"]
    df = df.merge(flights_df_50, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)

    flights_df_40 = flights_df.groupby(by=["beforeScheduled_40", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_40.rename(columns={'flight_ID': "flightNum_40", 'beforeScheduled_40':'slice10min'}, inplace=True)
    # flights_df_40["slice10min"] = flights_df_40["beforeScheduled_40"].apply(
    #    lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # del flights_df_40["beforeScheduled_40"]
    df = df.merge(flights_df_40, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)

    flights_df_30 = flights_df.groupby(by=["beforeScheduled_30", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_30.rename(columns={'flight_ID': "flightNum_30", 'beforeScheduled_30':'slice10min'}, inplace=True)
    # flights_df_30["slice10min"] = flights_df_30["beforeScheduled_30"].apply(
    #    lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # del flights_df_30["beforeScheduled_30"]
    df = df.merge(flights_df_30, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)

    flights_df_20 = flights_df.groupby(by=["beforeScheduled_20", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_20.rename(columns={'flight_ID': "flightNum_20", 'beforeScheduled_20':'slice10min'}, inplace=True)
    # flights_df_20["slice10min"] = flights_df_20["beforeScheduled_20"].apply(
    #    lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # del flights_df_20["beforeScheduled_20"]
    df = df.merge(flights_df_20, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)

    flights_df_10 = flights_df.groupby(by=["beforeScheduled_10", "BGATE_AREA"])["flight_ID"].count().reset_index()
    flights_df_10.rename(columns={'flight_ID': "flightNum_10", 'beforeScheduled_10':'slice10min'}, inplace=True)
    # flights_df_10["slice10min"] = flights_df_10["beforeScheduled_10"].apply(
    #    lambda x: str(x)[0:10] + "-" + str(x)[11:13] + "-" + str(x)[14:15])
    # del flights_df_10["beforeScheduled_10"]
    df = df.merge(flights_df_10, how="left", on=['BGATE_AREA', 'slice10min']).fillna(0)


    # flights_df_60.to_csv("clean_data/flights_df_60.csv", index=False)

    features_x.append("flightNum_60")
    features_x.append("flightNum_50")
    features_x.append("flightNum_40")
    features_x.append("flightNum_30")
    features_x.append("flightNum_20")
    features_x.append("flightNum_10")
    # 删除掉了11号的数据
    # df = df.loc[df["slice10min"].apply(lambda x: x[8:10]).isin(["12","13","14"]), features_x]


    # -------------------------------
    #  Add the security-check features
    # -------------------------------
    sc_df = pd.read_csv("../input/airport_gz_security_check_chusai_1stround.csv")
    sc_df["day"] = sc_df["security_time"].apply(lambda x: x.split(" ")[0])
    # sc_df = sc_df.groupby(by=["flight_ID", "security_time"]).apply(lambda x: x.reset_index())
    # del sc_df["index"]
    flights_df_org["day"] = flights_df_org["scheduled_flt_time"].apply(lambda x: str(x).split(" ")[0])
    sc_df = sc_df.merge(flights_df_org, how='left', on=["flight_ID", "day"]).fillna(0)
    del sc_df["scheduled_flt_time"]
    del sc_df["actual_flt_time"]
    sc_df = sc_df.merge(bgate_df_org, how='left', on=["BGATE_ID"]).fillna(0)
    sc_df["slice10min"] = sc_df["security_time"].apply(lambda x: parse(x).strftime('%Y-%m-%d-%H-%M'))
    sc_df["slice10min"] = sc_df["slice10min"].apply(lambda x: str(x)[0:-1])
    sc_df.to_csv("clean_data/sc_df_org.csv", index=False)
    sc_df = sc_df.groupby(by=["BGATE_AREA", "slice10min"])["passenger_ID"].count().reset_index()
    sc_df["passenger_ID"] = sc_df["passenger_ID"] / 10
    sc_df.rename(columns={"passenger_ID":"security_num"}, inplace=True)
    sc_df.to_csv("clean_data/sc_df.csv", index=False)

    df = df.merge(sc_df, how='left', on=["BGATE_AREA", "slice10min"]).fillna(method="pad")
    features_x.append("security_num")

    return df, features_x, features_y

def model_solution(model_name="xgb"):
    print("model_solution run!")

    train_df = pd.read_csv("clean_data/test12-15.csv")
    # train_df = pd.read_csv("../input/WIFI_AP_Passenger_Records_chusai_1stround.csv")
    train_df = train_df.loc[~train_df["timeStamp"].apply(lambda x: x[0:10]).isin(["2016-09-10"]),
                            ["WIFIAPTag", "passengerCount", "timeStamp"]]# 把9月10号的数据去除掉
    train_df["slice10min"] = train_df["timeStamp"].apply(lambda x: x[0:15])
    train_df = train_df.groupby(by=["WIFIAPTag", "slice10min"])["passengerCount"].sum().reset_index()
    train_df["passengerCount"] = (train_df["passengerCount"]/10).round(2)
    # train_df.to_csv("clean_data/train_df.csv", index=False)

    # train_df = pd.read_csv("clean_data/train_df.csv")
    # Construct test df
    test_ids = train_df["WIFIAPTag"].drop_duplicates()
    test_ids = pd.DataFrame(test_ids, columns=['WIFIAPTag']).reset_index()
    del test_ids['index']
    test_ids['passengerCount'] = 0
    # print test_ids
    pre_time = pd.date_range(start="20160914120000", end="20160914145959", freq="10min")
    # pre_time = pd.date_range(start="20160914150000", end="20160914175959", freq="10min")
    pre_time = [w.strftime("%Y-%m-%d-%H-%M") for w in pre_time]
    pre_time = [w[:-1] for w in pre_time]
    pre_time = pd.DataFrame(pre_time, columns=['slice10min'])
    pre_time['passengerCount'] = 0
    # print pre_time
    test_df = pd.merge(test_ids, pre_time, how='left', on='passengerCount')
    # print test_df
    test_df.to_csv("clean_data/test_df.csv", index=False)

    # 提取特征
    df, features_x, features_y = extract_features(train_df, test_df)  # 提取特征
    df.to_csv("clean_data/df.csv", index=False)
    # 将WIFIAPTag和slice10min分别替换成数字，用字典表示
    '''
    df["slice10min"] = df["slice10min"].apply(lambda x: int(x[8:10] + x[11:13] + x[-1:]))
    pre_WIFIAPTag = df["WIFIAPTag"].drop_duplicates().reset_index()
    del pre_WIFIAPTag["index"]
    # print pre_WIFIAPTag
    WIFIAPTag_dict = {}
    for i in range(len(pre_WIFIAPTag)):
        WIFIAPTag_dict[pre_WIFIAPTag["WIFIAPTag"][i]] = WIFIAPTag_dict.get(pre_WIFIAPTag["WIFIAPTag"][i], 0) + i
    # print WIFIAPTag_dict
    df["WIFIAPTag"] = df["WIFIAPTag"].apply(lambda x: WIFIAPTag_dict.get(x))
    '''
    # 使用sklearn的preprocessing.LabelEncoder方法把df中的字符串变成顺序的数字
    le = LabelEncoder()
    le.fit(df["WIFIAPTag"])
    df["WIFIAPTag"] = le.transform(df["WIFIAPTag"])
    le.fit(df["slice10min"])
    df["slice10min"] = le.transform(df["slice10min"])


    # print features_x
    X_train, X_val, y_train, y_val = train_test_split(df.loc[df["is_train"] == 1][features_x],
                                                      df.loc[df["is_train"] == 1][features_y],
                                                      test_size=0.01, random_state=42)
    X_test = df.loc[df["is_train"] == 0, features_x]  # 建立测试集
    if model_name == "xgb":  # 通过参数构造模型
        model = XGBOOST(X_train, y_train, X_val, y_val)
    elif model_name == "rf":
        model = RF(X_train, y_train, X_val, y_val)

    y_pre = model.guess(X_test)
    print y_pre

    result_df = test_df[["WIFIAPTag", "slice10min"]].reset_index()
    del result_df["index"]
    if model_name == "xgb":
        passengerCount = (np.expm1(y_pre) * 1).astype(np.int)
        result_df.insert(0, 'passengerCount', passengerCount)
        # result_df["passengerCount"] = (np.expm1(y_pre) * 1).astype(np.int)

    elif model_name == "rf":
        passengerCount = (np.expm1(y_pre) * 1.5).astype(np.int)
        result_df.insert(0, 'passengerCount', passengerCount)

    return result_df

def test(result):
    test_result = pd.read_csv("clean_data/test12-15_result.csv")
    test_result.rename(columns={"passengerCount":"passengerCount_result"}, inplace=True)
    test_result = test_result.merge(result, how='left', on=["WIFIAPTag", "slice10min"]).fillna(0)
    # test_result.to_csv("clean_data/test_result.csv", index=False)
    # Criteria = sum((result['passengerCount']-test_result['passengerCount'])*(result['passengerCount']-test_result['passengerCount']))
    # score = sum((test_result["passengerCount"] - test_result["passengerCount_result"])*(test_result["passengerCount"] - test_result["passengerCount_result"]))
    score = np.sum((test_result["passengerCount"].values - test_result["passengerCount_result"].values)**2)

    print("score = %f" % score)

if __name__ == "__main__":
    time0 = time.time()

    # 测试集
    '''
    filename = "clean_data/test12-15.csv"
    start_time = "20160925120000"
    end_time = "20160925145959"
    '''
    # 真正训练集

    filename = "../input/WIFI_AP_Passenger_Records_chusai_2ndround.csv"
    start_time = "20160925150000"
    end_time = "20160925175959"



    rule_result = rule_solution(filename, start_time, end_time)
    rule_result.to_csv("submissions/airport_gz_passenger_predict_rule_mean.csv", index=False)

    # model1_result = model_solution("xgb")
    # model1_result.to_csv("submissions/airport_gz_passenger_predict_model_xgb_2.csv", index=False)

    # model2_result = model_solution("rf")
    # model2_result.to_csv("submissions/airport_gz_passenger_predict_model_rf.csv", index=False)

    # test(rule_result)
    time1 = time.time()
    print ("It take %f s to process" % (time1 - time0))