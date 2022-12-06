# encoding: utf-8

import os
from data import fetch_data
from feature import gen_corpus
from model import train
from backtest import back_test

if __name__ == '__main__':
    fetch_data('20220101', '20221130')  # 调tushare，获取一段时间内所有股票的交易数据
    gen_corpus()  # 生成训练样本
    train()  # 训练模型
    back_test('20220920', '20221120', os.path.join("file", "record.csv"))  # 回测
