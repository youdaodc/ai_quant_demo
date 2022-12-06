# encoding: utf-8

import os
import pandas as pd
import lightgbm as lgb
from feature import FEATURE_DIR, LABEL_NAME

MODEL_FILE = os.path.join("file", "model", "lgb.txt")


def train():
    '''
    训练lightGBM(决策树)模型
    '''
    train_corpus_list, test_corpus_list = [], []
    for file in os.listdir(FEATURE_DIR):
        df = pd.read_pickle(os.path.join(FEATURE_DIR, file))  # 从文件中加载样本数据
        pivot = int(len(df) * 0.6)  # 训练集和测试集六四开
        train_corpus_list.append(df[:pivot])
        test_corpus_list.append(df[pivot:])
    train_corpus = pd.concat(train_corpus_list, axis=0)
    test_corpus = pd.concat(test_corpus_list, axis=0)

    feature_names = [ele for ele in train_corpus.columns if ele != LABEL_NAME]
    train_x = train_corpus.loc[:, feature_names]
    train_y = train_corpus.loc[:, LABEL_NAME]
    test_x = test_corpus.loc[:, feature_names]
    test_y = test_corpus.loc[:, LABEL_NAME]

    # lightGBM默认把Nan当作缺失值
    dtrain = lgb.Dataset(data=train_x, label=train_y, feature_name=feature_names,
                         params={'num_threads': 8, 'use_missing': True, 'zero_as_missing': False, 'verbose': 0}, )
    dtest = lgb.Dataset(data=test_x, label=test_y, feature_name=feature_names,
                        params={'num_threads': 8, 'use_missing': True, 'zero_as_missing': False, 'verbose': 0})
    gbm = lgb.train(params={'learning_rate': 0.01,  # 学习率
                            'max_depth': 3,  # 每棵树的最大深度。特征越多，深度应该越大
                            },
                    num_boost_round=10,  # 多少棵树
                    train_set=dtrain, valid_sets=[dtrain, dtest],
                    )
    # 保存模型文件，文件末尾记录了各特征的重要度
    gbm.save_model(MODEL_FILE)


if __name__ == '__main__':
    train()
