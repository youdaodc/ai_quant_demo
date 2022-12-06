# encoding: utf-8

import os

import pandas as pd
from feature import FEATURE_DIR, LABEL_NAME
from model import MODEL_FILE
from datetime import datetime, timedelta
import lightgbm as lgb
import numpy as np


def back_test(start_date, end_date, outfile):
    '''
    回测
    :param start_date: 回测开始日志，str类型，%Y%m%d格式，包含当天
    :param end_date: 回测结束日志，str类型，%Y%m%d格式，包含当天
    :param outfile: 输出文件，记录每天买哪支股票以及收益
    '''
    df_list = []
    for file in os.listdir(FEATURE_DIR):
        df = pd.read_pickle(os.path.join(FEATURE_DIR, file))
        df_list.append(df)
    corpus = pd.concat(df_list, axis=0)  # 所有股票的特征合在一起
    dates = set(corpus.index.get_level_values('trade_date').unique().values.tolist())  # 取得所有的交易日

    fout = open(outfile, "w")
    fout.write("date,symbole,score,rise,cum\n")
    cum = 0
    model = lgb.Booster(model_file=MODEL_FILE)  # 从文件中加载lightGBM模型
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    returns = []
    while end_date >= start_date:
        date = int(start_date.strftime("%Y%m%d"))
        if date in dates:  # 不在dates里的日期不是正常交易日
            data = corpus.loc[date]
            y_hat = model.predict(data=data.values, predict_disable_shape_check=True)  # 用模型预测涨幅
            predict_result = pd.Series(y_hat, index=data.index)  # 预测score跟股票代码对应起来
            sorted_stock = predict_result.sort_values(ascending=False)  # 按score降序排列
            for symbol, score in sorted_stock.iteritems():
                rise = data.loc[symbol][LABEL_NAME]
                cum += rise
                fout.write(",".join(map(str, [date, symbol, score, rise, cum])) + "\n")
                returns.append(rise)
                break  # 只取score最高的那个
        start_date += timedelta(days=1)
    fout.close()
    cum, md = max_drawdown(returns)
    print(
        f"累计收益率{100 * cum:.2f}%, 最大回撤率{100 * md:.2f}%, 夏普率{100 * sharp_ratio(returns):.2f}%")


def max_drawdown(return_list):
    '''
    最大回撤率
    :param return_list: 每日收益率
    :return:累计收益率 和 最大回撤率
    '''
    cum = 0
    highest = 0  # 最高点
    max_draw_down = 0  # 最大回撤
    max_draw_down_list = list()
    for ele in return_list:
        cum += ele
        if cum > highest:
            highest = cum
            if max_draw_down > 0:
                max_draw_down_list.append(max_draw_down)
                max_draw_down = 0
        else:
            draw_down = (highest - cum) / highest
            if draw_down > max_draw_down:
                max_draw_down = draw_down
    if max_draw_down > 0:
        max_draw_down_list.append(max_draw_down)
    return cum, max(max_draw_down_list)


def sharp_ratio(return_list):
    '''
    夏普率
    :param return_list: 每日收益率
    :return: 夏普率
    '''
    mean = np.mean(return_list)
    std = np.std(return_list)
    return mean / std


if __name__ == '__main__':
    action_file = os.path.join("file", "record.csv")
    back_test('20220915', '20221115', action_file)
