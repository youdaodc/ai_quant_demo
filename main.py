# encoding: utf-8

import os
from data import fetch_data
from feature import gen_corpus
from model import train
from backtest import back_test

if __name__ == '__main__':
    fetch_data('20220101', '20221130')
    gen_corpus()
    train()
    back_test('20220920', '20221120', os.path.join("file", "record.csv"))
