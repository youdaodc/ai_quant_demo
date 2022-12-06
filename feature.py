# encoding: utf-8

import os

import talib as tb
import pandas as pd
from data import OHLCV_DIR, ALL_STOCK_LIST

LABEL_NAME = "LABEL"
FEATURE_DIR = os.path.join("file", "feature")


def cal_label(df):
    '''
    计算机器学习模型的拟合目标
    :param df: DataFrame，某支股票的历史OHLCV数据包含在其中
    :return: Series
    '''
    open = df['open']
    buy_price = open.shift(-1)  # 特征是在当天收盘后计算好的，以次日开盘价买入
    sell_price = open.shift(-2)  # 以次次日开盘价卖入
    rise = sell_price / buy_price - 1  # 涨幅，作为AI模型拟合的目标
    label = pd.Series(rise, index=open.index)
    label.rename(LABEL_NAME, inplace=True)
    return label


def cal_feature(df):
    '''
    根据一支股票的历史数据，计算高级特征。
    注意：特征是在当天收盘后计算好的。
    :param df: DataFrame，某支股票的历史OHLCV数据包含在其中
    :return: Series
    '''
    close, high, low = df["close"], df["high"], df["low"]

    '''当前价格、短期平滑的价格、长期平滑的价格，这3者之间互相求差'''
    EMA_short = tb.EMA(close, timeperiod=5)  # 短期EMA平滑曲线，快线
    EMA_long = tb.EMA(close, timeperiod=20)  # 长期EMA平滑曲线，慢线
    f1 = (close - EMA_short) / close  # 除以close是为了在各股票之间统一量纲
    f1.rename("CLOSE_EMA_SHORT", inplace=True)
    f2 = (close - EMA_long) / close
    f2.rename("CLOSE_EMA_LONG", inplace=True)
    f3 = (EMA_short - EMA_long) / close
    f3.rename("EMA_SHORT_LONG", inplace=True)

    '''布林带，计算当前价格到上限、下限、中线的距离'''
    BOLLINGER_UPPER, BOLLINGER_MIDDLE, BOLLINGER_LOWER = tb.BBANDS(close)  # 布林带，根据股价的标准差和置信区间，确定股价的波动范围
    f4 = (close - BOLLINGER_UPPER) / close  # 除以close是为了在各股票之间统一量纲
    f4.rename("CLOSE_BOLLINGER_UPPER", inplace=True)
    f5 = (close - BOLLINGER_MIDDLE) / close
    f5.rename("CLOSE_BOLLINGER_MIDDLE", inplace=True)
    f6 = (close - BOLLINGER_LOWER) / close
    f6.rename("CLOSE_BOLLINGER_LOWER", inplace=True)

    '''PPO(MACD的归一化版本)'''
    f7 = tb.PPO(close, fastperiod=12, slowperiod=26)
    f7.rename("PPO", inplace=True)

    feature_df = pd.concat([f1, f2, f3, f4, f5, f6, f7], axis=1)
    return feature_df


def gen_corpus():
    '''
    针对每一支股票，计算feature和label，结果写入对应的文件
    '''
    for symbol in ALL_STOCK_LIST:
        df = pd.read_csv(os.path.join(OHLCV_DIR, symbol))
        df['symbol'] = symbol
        df.set_index(['trade_date', 'symbol'], inplace=True)  # 日期和股票代码共同作为index
        feature = cal_feature(df)
        label = cal_label(df)
        corpus = pd.concat([feature, label], axis=1)
        corpus.to_pickle(os.path.join(FEATURE_DIR, symbol))  # 结果写入对应的文件，以股票代码作为文件名


if __name__ == '__main__':
    gen_corpus()
