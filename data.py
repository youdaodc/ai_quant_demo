# encoding: utf-8

import os
import tushare as ts

ts_token = open(os.path.join("file", "tushare_token.txt")).readline().strip()  # 从文件中读取你的tushare token
ts.set_token(ts_token)

ALL_STOCK_LIST = [ele.strip() for ele in
                  open(os.path.join("file", "stock_list.txt")).readlines()]  # 所有股票的集合（文件里的内容根据自己的需求定制）
OHLCV_DIR = os.path.join("file", "data")  # 此文件中存放所有股票的open/high/low/close/volume数据


def fetch_data(start_date, end_date):
    '''
    调tushare获取所有股票的历史价格
    :param start_date: 历史开始日期，str类型，%Y%m%d格式，包含当天
    :param end_date: 历史结束日期，str类型，%Y%m%d格式，包含当天
    :return:
    '''
    for symbol in ALL_STOCK_LIST:
        df = ts.pro_bar(  # 关于pro_bar函数的参数说明参见 https://tushare.pro/document/2?doc_id=109
            adj='hfq',  # 量化模型一般都使用后复权
            ts_code=symbol, start_date=start_date, end_date=end_date)
        df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol']]  # 只选取所需要的列
        df.sort_values(by='trade_date', inplace=True)  # 按日期升序排列（默认是按日期降序排）
        df.to_csv(os.path.join(OHLCV_DIR, symbol), index=False)  # 价格数据写入文件，文件名就是股票代码


if __name__ == '__main__':
    fetch_data('20220101', '20221130')  # 获取2022年前11个月的数据
