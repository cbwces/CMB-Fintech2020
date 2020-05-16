#!/usr/bin/env python
# coding: utf-8

import gc
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from scipy import stats
from gensim.models import Word2Vec

# 传递参数
parser = argparse.ArgumentParser()
parser.add_argument("--TRAIN_PATH", type=str, help="训练集路径")
parser.add_argument("--TEST_PATH", type=str, help='测试集路径')
parser.add_argument("--FEATURE_PATH", type=str, help='特征路径')
args = parser.parse_args()
train_p = "."
test_p = "."
feature_p = "."
if args.TRAIN_PATH:
    train_p = args.TRAIN_PATH
if args.TEST_PATH:
    test_p = args.TEST_PATH
if args.FEATURE_PATH:
    feature_p = args.FEATURE_PATH

def reduce_mem_usage(df):
    # 减小DataFrame内存开销
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# 求一列特征众数以及相应的出现次数
def get_mode(x):
    mode = stats.mode(x)[0][0]
    return mode

def get_mode_freq(x):
    md = get_mode(x)
    return np.where(x == md)[0].shape[0]
# 

# Word2Vec相关函数
def get_sentence(x, container):
    # 获取W2V训练预料 
    container.append(x.astype(str).values.tolist())
    return True

def trad_vec(x, uid, words):
    # W2V特征——向量转化，同一uid各词向量结果求平均
    uid.append(x.name)
    vc = np.zeros(6)
    for i in x.values:
        vc += word_model.wv[i]
    vc /= len(x)
    words.append(vc)
    return True

# 读取数据于基本转换
train_user = pd.read_csv(train_p+"/训练数据集_tag.csv")
train_beh = pd.read_csv(train_p+"/训练数据集_beh.csv")
train_trd = pd.read_csv(train_p+"/训练数据集_trd.csv")
test_user= pd.read_csv(test_p+"/评分数据集_tag.csv")
test_beh= pd.read_csv(test_p+"/评分数据集_beh.csv")
test_trd= pd.read_csv(test_p+"/评分数据集_trd.csv")

train_beh['page_tm'] = pd.to_datetime(train_beh['Unnamed: 3'])
test_beh['page_tm'] = pd.to_datetime(test_beh['Unnamed: 2'])
del train_beh['Unnamed: 3'], test_beh['Unnamed: 2']
train_trd['trx_tm'] = pd.to_datetime(train_trd['trx_tm'])
test_trd['trx_tm'] = pd.to_datetime(test_trd['trx_tm'])

# 时间排序整理
train_beh['page_tm'] = train_beh.groupby("id")['page_tm'].transform(lambda x: sorted(x.values))
test_beh['page_tm'] = test_beh.groupby("id")['page_tm'].transform(lambda x: sorted(x.values))
train_trd['trx_tm'] = train_trd.groupby("id")['trx_tm'].transform(lambda x: sorted(x.values))
test_trd['trx_tm'] = test_trd.groupby("id")['trx_tm'].transform(lambda x: sorted(x.values))

# trd表和beh表各同一用户各记录时间差
train_trd_diff = train_trd.groupby('id', sort=False)['trx_tm'].diff().dt.components
train_trd_diff['id'] = train_trd.id
test_trd_diff = test_trd.groupby('id', sort=False)['trx_tm'].diff().dt.components
test_trd_diff['id'] = test_trd.id

train_beh_diff = train_beh.groupby('id', sort=False)['page_tm'].diff().dt.components
train_beh_diff['id'] = train_beh.id
test_beh_diff = test_beh.groupby('id', sort=False)['page_tm'].diff().dt.components
test_beh_diff['id'] = test_beh.id

# 按题理解筛选类别特征
catagrical_features = ["crd_card_act_ind", "atdd_type", "gdr_cd",
                 "mrg_situ_cd", "edu_deg_cd", "ic_ind", "fr_or_sh_ind", "dnl_mbl_bnk_ind", 
                 "dnl_bind_cmb_lif_ind", "hav_car_grp_ind", "hav_hou_grp_ind", "l6mon_agn_ind", 
                 "vld_rsk_ases_ind", "loan_act_ind", "acdm_deg_cd", "deg_cd", 'bk1_cur_year_mon_avg_agn_amt_cd', 
                      'l1y_crd_card_csm_amt_dlm_cd', 'l6mon_daim_aum_cd', 'perm_crd_lmt_cd','pl_crd_lmt_cd']

# 数据清洗
for feature in [c for c in train_user.columns if c not in ['id', 'flag']]:
    for value in set(train_user[feature].unique().append(test_user[feature].unique())):
        if value in ['\\N', "31", "30"]:
            if "-1" not in train_user[feature].unique():
                train_user[feature] = train_user[feature].replace(['\\N', "31", "30"], np.nan)
                test_user[feature] = test_user[feature].replace(['\\N', "31", "30"], np.nan)
            else:
                if feature in catagrical_features:
                    train_user[feature] = train_user[feature].replace(["-1", '\\N', "31", "30"], np.nan)
                    test_user[feature] = test_user[feature].replace(["-1", '\\N', "31", "30"], np.nan)
                else:
                    train_user[feature] = train_user[feature].replace(['\\N', "31", "30"], np.nan).astype(float)
                    test_user[feature] = test_user[feature].replace(['\\N', "31", "30"], np.nan).astype(float)
                    if (train_user[feature].min() == -1) or (test_user[feature].min() == -1):
                        train_user[feature] = train_user.replace(-1, np.nan)
                        test_user[feature] = test_user.replace(-1, np.nan)

# 统计时间相关特征
for beh in [train_beh, test_beh]:
    beh['day'] = beh['page_tm'].dt.dayofyear
    beh['week'] = beh['page_tm'].dt.weekofyear
    beh['weekday'] = beh['page_tm'].dt.dayofweek
    beh['hour'] = beh['page_tm'].dt.hour

for trd in [train_trd, test_trd]:
    trd['day'] = trd['trx_tm'].dt.dayofyear
    trd['week'] = trd['trx_tm'].dt.weekofyear
    trd['weekday'] = trd['trx_tm'].dt.dayofweek
    trd['hour'] = trd['trx_tm'].dt.hour

train_user = train_user.merge(train_trd_diff.groupby('id')['days'].max().rename('max_day_diff'), on='id', how='left')
train_user = train_user.merge(train_trd_diff.groupby('id')['days'].mean().rename('mean_day_diff'), on='id', how='left')
train_user = train_user.merge(train_trd_diff.groupby('id')['days'].std().rename('std_day_diff'), on='id', how='left')

test_user = test_user.merge(test_trd_diff.groupby('id')['days'].max().rename('max_day_diff'), on='id', how='left')
test_user = test_user.merge(test_trd_diff.groupby('id')['days'].mean().rename('mean_day_diff'), on='id', how='left')
test_user = test_user.merge(test_trd_diff.groupby('id')['days'].std().rename('std_day_diff'), on='id', how='left')

train_user = train_user.merge(train_beh_diff.groupby('id')['days'].max().rename('max_day_diff2'), on='id', how='left')
train_user = train_user.merge(train_beh_diff.groupby('id')['days'].mean().rename('mean_day_diff2'), on='id', how='left')
train_user = train_user.merge(train_beh_diff.groupby('id')['days'].std().rename('std_day_diff2'), on='id', how='left')

test_user = test_user.merge(test_beh_diff.groupby('id')['days'].max().rename('max_day_diff2'), on='id', how='left')
test_user = test_user.merge(test_beh_diff.groupby('id')['days'].mean().rename('mean_day_diff2'), on='id', how='left')
test_user = test_user.merge(test_beh_diff.groupby('id')['days'].std().rename('std_day_diff2'), on='id', how='left')

# word2vec特征
words_sent = []
train_trd.groupby('id')["Trx_Cod2_Cd"].apply(lambda x: get_sentence(x, words_sent))
test_trd.groupby('id')["Trx_Cod2_Cd"].apply(lambda x: get_sentence(x, words_sent))
word_model = Word2Vec(words_sent, size=6, window=5, min_count=1, workers=8)

uud = []
wordser = [] 
train_trd.groupby("id")['Trx_Cod2_Cd'].apply(lambda x: trad_vec(x.astype(str), uud, wordser))
train_word_df = pd.DataFrame(np.array(wordser))
train_word_df['id'] = uud
uud = []
wordser = []
test_trd.groupby("id")['Trx_Cod2_Cd'].apply(lambda x: trad_vec(x.astype(str), uud, wordser))
test_word_df = pd.DataFrame(np.array(wordser))
test_word_df['id'] = uud

train_user = train_user.merge(train_word_df, on="id", how='left')
test_user = test_user.merge(test_word_df, on="id", how='left')

# 交叉列特征构造 
feature_stock = ['l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'l12_mon_gld_buy_whl_tms']
train_user['l12mon_buy_sum'] = train_user[feature_stock].sum(axis=1)
test_user['l12mon_buy_sum'] = train_user[feature_stock].sum(axis=1)

for col in feature_stock:
    train_user[f'{col}_ratio'] = train_user[col].astype(float) / (train_user['l12mon_buy_sum']+1)
    test_user[f'{col}_ratio'] = test_user[col].astype(float) / (test_user['l12mon_buy_sum']+1)    

train_trd = train_trd.merge(train_trd.groupby(["id", "week"])['trx_tm'].count().rename('trx_tm_week'), on=['id', 'week'], how='left')
train_trd = train_trd.merge(train_trd.groupby(["id", "weekday"])['trx_tm'].count().rename('trx_tm_weekday'), on=['id', 'weekday'], how='left')

test_trd = test_trd.merge(test_trd.groupby(["id", "week"])['trx_tm'].count().rename('trx_tm_week'), on=['id', 'week'], how='left')
test_trd = test_trd.merge(test_trd.groupby(["id", "weekday"])['trx_tm'].count().rename('trx_tm_weekday'), on=['id', 'weekday'], how='left')

train_user["cur_debit_rate"] = train_user.cur_debit_min_opn_dt_cnt.astype(float) * train_user.cur_debit_cnt.astype(float)
train_user["cur_credit_rate"] = train_user.cur_credit_min_opn_dt_cnt.astype(float) * train_user.cur_credit_cnt.astype(float)

train_user["creditrate1"] = train_user.l1y_crd_card_csm_amt_dlm_cd.astype(float) * train_user.cur_credit_min_opn_dt_cnt.astype(float)
train_user["creditrate2"] = train_user.perm_crd_lmt_cd.astype(float) * train_user.cur_credit_min_opn_dt_cnt.astype(float)
train_user["creditrate3"] = train_user.l1y_crd_card_csm_amt_dlm_cd.astype(float) * train_user.perm_crd_lmt_cd.astype(float)

test_user["cur_debit_rate"] = test_user.cur_debit_min_opn_dt_cnt.astype(float) * test_user.cur_debit_cnt.astype(float)
test_user["cur_credit_rate"] = test_user.cur_credit_min_opn_dt_cnt.astype(float) * test_user.cur_credit_cnt.astype(float)

test_user["creditrate1"] = test_user.l1y_crd_card_csm_amt_dlm_cd.astype(float) * test_user.cur_credit_min_opn_dt_cnt.astype(float)
test_user["creditrate2"] = test_user.perm_crd_lmt_cd.astype(float) * test_user.cur_credit_min_opn_dt_cnt.astype(float)
test_user["creditrate3"] = test_user.l1y_crd_card_csm_amt_dlm_cd.astype(float) * test_user.perm_crd_lmt_cd.astype(float)

# 按“Dat_Flag3_Cd”分组计算交易金额的统计特征, 按"Dat_Flag3_Cd"对应的3个值以及对应金额的6项统计做18列新特征
trx2_train_df = train_trd.groupby(['id', 'Dat_Flg3_Cd'])['cny_trx_amt'].agg(['max', 'min', 'median', 'std', 'mean', 'sum']).reset_index()
for trx2 in ["A", "B", "C"]:
    train_user = train_user.merge(trx2_train_df.loc[trx2_train_df.Dat_Flg3_Cd == trx2, ['id', 'max', 'min', 'median', 'std', 'mean', 'sum']].rename({'max': f'{str(trx2)}_max', 'min': f'{str(trx2)}_min', 'median': f'{str(trx2)}_median',             'std': f'{str(trx2)}_std', 'mean': f'{str(trx2)}_mean', 'sum': f'{str(trx2)}_sum'}, axis=1), on='id', how='left')
for v in train_user.columns[-18:]:
    train_user.loc[train_user.id.isin(train_trd.id.values), v] = train_user.loc[train_user.id.isin(train_trd.id.values), v].fillna(0)

trx2_test_df = test_trd.groupby(['id', 'Dat_Flg3_Cd'])['cny_trx_amt'].agg(['max', 'min', 'median', 'std', 'mean', 'sum']).reset_index()
for trx2 in ["A", "B", "C"]:
    test_user = test_user.merge(trx2_test_df.loc[trx2_test_df.Dat_Flg3_Cd == trx2, ['id', 'max', 'min', 'median', 'std', 'mean', 'sum']].rename({'max': f'{str(trx2)}_max', 'min': f'{str(trx2)}_min', 'median': f'{str(trx2)}_median',             'std': f'{str(trx2)}_std', 'mean': f'{str(trx2)}_mean', 'sum': f'{str(trx2)}_sum'}, axis=1), on='id', how='left',)
for v in test_user.columns[-18:]:
    test_user.loc[test_user.id.isin(test_trd.id.values), v] = test_user.loc[test_user.id.isin(test_trd.id.values), v].fillna(0)

# "Dat_Flg1_Cd"只有两个值，转换为01编码便于后续统计平均，并对trd出现的交易记录特征做统计，合并入主表
train_trd['Dat_Flg1_Cd'] = train_trd['Dat_Flg1_Cd'].map({"B": 0, "C": 1})
test_trd['Dat_Flg1_Cd'] = test_trd['Dat_Flg1_Cd'].map({"B": 0, "C": 1})

train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg1_Cd'].count().rename("dat1_count"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg1_Cd'].nunique().rename("dat1_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg1_Cd'].mean().rename("dat1_mean"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg1_Cd'].agg(lambda x: get_mode_freq(x)).rename("dat1_modev"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg3_Cd'].nunique().rename("dat3_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg3_Cd'].agg(lambda x: get_mode(x)).rename("dat3_mode"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Dat_Flg3_Cd'].agg(lambda x: get_mode_freq(x)).rename("dat3_modev"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod1_Cd'].nunique().rename("trx1_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod1_Cd'].agg(lambda x: get_mode(x)).rename("trx1_mode"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod1_Cd'].agg(lambda x: get_mode_freq(x)).rename("trx1_modev"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod2_Cd'].nunique().rename("trx2_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod2_Cd'].agg(lambda x: get_mode(x)).rename("trx2_mode"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['Trx_Cod2_Cd'].agg(lambda x: get_mode_freq(x)).rename("trx2_modev"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['cny_trx_amt'].min().rename("cny_trx_amt_min"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['cny_trx_amt'].max().rename("cny_trx_amt_max"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_mean"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_median"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_sum"), on="id", how="left")

train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].nunique().rename("cny_trx_amt_positive_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].min().rename("cny_trx_amt_positive_min"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_positive_mean"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_positive_median"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].std().rename("cny_trx_amt_positive_std"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_positive_sum"), on="id", how="left")

train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].nunique().rename("cny_trx_amt_negative_nunique"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].max().rename("cny_trx_amt_negative_max"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_negative_mean"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_negative_median"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].std().rename("cny_trx_amt_negative_std"), on="id", how="left")
train_user = train_user.merge(train_trd.loc[train_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_negative_sum"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['week'].std().rename("week_std2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['week'].nunique().rename("week_nunique2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['week'].agg(lambda x: get_mode(x)).rename("week_mode2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['week'].median().rename("week_median2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['week'].mean().rename("week_mean2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['week'].agg(lambda x: get_mode_freq(x)).rename("week_mode2v"), on="id", how="left")
train_user = train_user.merge((train_trd.groupby('id')['week'].mean() - train_trd.groupby('id')['week'].median()).rename("week_kurl2"), on="id", how="left")
train_user = train_user.merge((train_trd.groupby('id')['week'].max() - train_trd.groupby('id')['week'].min()).rename("week_mm2"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['weekday'].nunique().rename("weekday_nunique2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['weekday'].agg(lambda x: get_mode(x)).rename("weekday_mode2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['weekday'].agg(lambda x: get_mode_freq(x)).rename("weekday_mode2v"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['hour'].nunique().rename("hour_nunique2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['hour'].agg(lambda x: get_mode_freq(x)).rename("hour_mode2v"), on="id", how="left")

test_trd['Dat_Flg1_Cd'] = test_trd['Dat_Flg1_Cd'].map({"B": 0, "C": 1})
test_trd['Dat_Flg1_Cd'] = test_trd['Dat_Flg1_Cd'].map({"B": 0, "C": 1})

test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg1_Cd'].count().rename("dat1_count"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg1_Cd'].nunique().rename("dat1_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg1_Cd'].mean().rename("dat1_mean"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg1_Cd'].agg(lambda x: get_mode_freq(x)).rename("dat1_modev"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg3_Cd'].nunique().rename("dat3_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg3_Cd'].agg(lambda x: get_mode(x)).rename("dat3_mode"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Dat_Flg3_Cd'].agg(lambda x: get_mode_freq(x)).rename("dat3_modev"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod1_Cd'].nunique().rename("trx1_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod1_Cd'].agg(lambda x: get_mode(x)).rename("trx1_mode"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod1_Cd'].agg(lambda x: get_mode_freq(x)).rename("trx1_modev"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod2_Cd'].nunique().rename("trx2_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod2_Cd'].agg(lambda x: get_mode(x)).rename("trx2_mode"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['Trx_Cod2_Cd'].agg(lambda x: get_mode_freq(x)).rename("trx2_modev"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['cny_trx_amt'].min().rename("cny_trx_amt_min"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['cny_trx_amt'].max().rename("cny_trx_amt_max"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_mean"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_median"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_sum"), on="id", how="left")

test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].nunique().rename("cny_trx_amt_positive_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].min().rename("cny_trx_amt_positive_min"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_positive_mean"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_positive_median"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].std().rename("cny_trx_amt_positive_std"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt > 0, :].groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_positive_sum"), on="id", how="left")

test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].nunique().rename("cny_trx_amt_negative_nunique"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].max().rename("cny_trx_amt_negative_max"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].mean().rename("cny_trx_amt_negative_mean"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].median().rename("cny_trx_amt_negative_median"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].std().rename("cny_trx_amt_negative_std"), on="id", how="left")
test_user = test_user.merge(test_trd.loc[test_trd.cny_trx_amt < 0, :].groupby('id')['cny_trx_amt'].sum().rename("cny_trx_amt_negative_sum"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['week'].std().rename("week_std2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['week'].nunique().rename("week_nunique2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['week'].agg(lambda x: get_mode(x)).rename("week_mode2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['week'].median().rename("week_median2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['week'].mean().rename("week_mean2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['week'].agg(lambda x: get_mode_freq(x)).rename("week_mode2v"), on="id", how="left")
test_user = test_user.merge((test_trd.groupby('id')['week'].mean() - test_trd.groupby('id')['week'].median()).rename("week_kurl2"), on="id", how="left")
test_user = test_user.merge((test_trd.groupby('id')['week'].max() - test_trd.groupby('id')['week'].min()).rename("week_mm2"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['weekday'].nunique().rename("weekday_nunique2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['weekday'].agg(lambda x: get_mode(x)).rename("weekday_mode2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['weekday'].agg(lambda x: get_mode_freq(x)).rename("weekday_mode2v"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['hour'].nunique().rename("hour_nunique2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['hour'].agg(lambda x: get_mode_freq(x)).rename("hour_mode2v"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['trx_tm_week'].max().rename("week_max2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['trx_tm_week'].min().rename("week_min2"), on="id", how="left")

train_user = train_user.merge(train_trd.groupby('id')['trx_tm_weekday'].max().rename("weekday_max2"), on="id", how="left")
train_user = train_user.merge(train_trd.groupby('id')['trx_tm_weekday'].min().rename("weekday_min2"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['trx_tm_week'].max().rename("week_max2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['trx_tm_week'].min().rename("week_min2"), on="id", how="left")

test_user = test_user.merge(test_trd.groupby('id')['trx_tm_weekday'].max().rename("weekday_max2"), on="id", how="left")
test_user = test_user.merge(test_trd.groupby('id')['trx_tm_weekday'].min().rename("weekday_min2"), on="id", how="left")

# 众数特征在之后的模型中采用不同方式处理，因此单独列出
real_cat = ['dat3_mode', 'trx1_mode', 'trx2_mode', "week_mode2", 'weekday_mode2']

# 为lgb模型训练进行类别转换
for i in real_cat:
    if i in train_user.columns:
        train_user[i] = train_user[i].astype("category")
        test_user[i] = test_user[i].astype("category")

train_cp = train_user.copy()
test_cp = test_user.copy()

# 特征筛选,最终筛除50个左右特征,方法参考这位老哥: https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
# gain_feats:用于训练的所有特征
# gain_cat_feats:其中选择的类别特征
null_imp_df = pd.read_csv(feature_p+'/null_importances_distribution_rf.csv')
actual_imp_df = pd.read_csv(feature_p+'/actual_importances_ditribution_rf.csv')
correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))
corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
gain_feats = [_f for _f, _, _score in correlation_scores if _score >= 40]
gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= 40) & (_f in real_cat)]

# 合并表格，便于下列的特征构建
all_user = train_user[['id']+catagrical_features].append(test_user[['id']+catagrical_features])
all_trd = train_trd[['id', 'cny_trx_amt']].append(test_trd[['id', 'cny_trx_amt']]).append(test_trd[['id', 'cny_trx_amt']]).reset_index(drop=True)
all_trd = all_trd.merge(all_user.reset_index(drop=True), on='id', how='left')

# 统计标签个数，为下面woe模型特征准备
flag_1_count = train_cp['flag'].sum()
flag_0_count = len(train_cp) - flag_1_count

for c in catagrical_features:
    # woe特征构建，为防止leak只对出现最小类样本个数大于100的类别特征进行woe计算
    if train_cp[c].value_counts().min() > 100:
        train_cp[c+"_crossentroy"] = train_cp[c].map(train_cp.groupby(c)['flag'].apply(lambda x: np.log((x.sum() / flag_1_count)*(flag_0_count / (x.count()-x.sum())))))
        test_cp[c+"_crossentroy"] = test_cp[c].map(train_cp.groupby(c)['flag'].apply(lambda x: np.log((x.sum() / flag_1_count)*(flag_0_count / (x.count()-x.sum())))))

    # 类别的金额特征统计
    train_cp[c+"_cny_median"] = train_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"_cny_median"] = test_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"_cny_sum"] = train_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"_cny_sum"] = test_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].sum()).astype(float)
    
    train_cp[c+"pos_cny_median"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"pos_cny_median"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"pos_cny_sum"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"pos_cny_sum"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].sum()).astype(float)

    train_cp[c+"neg_cny_median"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"neg_cny_median"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"neg_cny_sum"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"neg_cny_sum"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].sum()).astype(float)

# 训练模型前的准备，将上述交易金额相关的统计特征合并到主表
train_cp.columns = [str(c) for c in train_cp.columns]
test_cp.columns = [str(c) for c in test_cp.columns]
entroy_col = []
for i in test_cp.columns:
    if "_crossentroy" in i:
        entroy_col.append(i)
st = train_user.id.isin(train_trd.id.values).astype(int)
train_cp = reduce_mem_usage(train_cp[gain_feats+entroy_col])
test_cp = reduce_mem_usage(test_cp[gain_feats+entroy_col])
gc.collect()

# lightgbm
for col in test_cp.columns:
    if col in gain_cat_feats:
        if col not in ['flag', 'id']:
            train_cp[col] = train_cp[col].astype('category')
            test_cp[col] = test_cp[col].astype('category')
    else:
        train_cp[col] = train_cp[col].astype('float32')
        test_cp[col] = test_cp[col].astype('float32')
lgb_params = {"objective": "binary", 
            #   "device_type": "gpu", 
              "n_jobs": -1, 
              "boosting": "gbdt",
              "seed": 999, 
              "metric": "auc", 
              'learning_rate': 0.01,
              'num_leaves': 300,
              'feature_fraction': 0.55,
              'bagging_fraction': 0.85,
              'bagging_freq': 15,
              'min_child_samples': 80,
             }
result = np.zeros((5, len(test_cp)))
offline_score1 = []
oof = np.zeros(len(train_cp))
feature_importance = np.zeros(train_cp.shape[1])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
for fold, (train_idx, test_idx) in enumerate(skf.split(train_cp[[c for c in train_cp.columns if c not in ["flag", "id"]]], st)):
    dtrainx, dtrainy = train_cp.loc[train_idx, [c for c in train_cp.columns if c not in ["flag", "id"]]], train_user.flag[train_idx]
    dvalidx, dvalidy = train_cp.loc[test_idx, [c for c in train_cp.columns if c not in ["flag", "id"]]], train_user.flag[test_idx]
    lgb_train = lgb.Dataset(dtrainx, label=dtrainy, categorical_feature=gain_cat_feats)
    lgb_valid = lgb.Dataset(dvalidx, label=dvalidy, categorical_feature=gain_cat_feats)
    lgb_model = lgb.train(lgb_params, 
              lgb_train, 
              num_boost_round=5500, 
              valid_sets=[lgb_train, lgb_valid], 
              early_stopping_rounds=200, 
              verbose_eval=200, 
#               callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.03*(np.e**(-0.005*x)))]
                         )
    offline_score1.append(lgb_model.best_score['valid_1']['auc'])
    oof[test_idx] = lgb_model.predict(dvalidx)
    feature_importance += lgb_model.feature_importance(importance_type="split")
    result[fold, :] = lgb_model.predict(test_cp[[c for c in test_cp.columns if c not in ["flag", "id"]]])
print("-" * 50)
print(offline_score1)
print("mean {} std {}".format(np.mean(offline_score1).round(6), np.std(offline_score1).round(6)))
np.save("./b_lgb_result", result.mean(axis=0))

# catboost
cat_list = []
for i, feat_name in enumerate([c for c in train_cp.columns if c not in ["id", "flag"]]):
    if feat_name in gain_cat_feats:
        cat_list.append(i)
for p in train_cp.columns:
    if p in gain_cat_feats:
        train_cp[p] = train_cp[p].astype("str")
        test_cp[p] = test_cp[p].astype("str")
cat_params = {"objective": "Logloss", 
              "eval_metric": "AUC", 
              "num_boost_round": 8000, 
              "learning_rate": 0.01, 
              "random_seed": 999, 
              "bootstrap_type": "Bernoulli", 
              "subsample": 0.84,
              "grow_policy": "Lossguide", 
              "max_leaves": 300,
              "max_depth": 9, 
              "min_data_in_leaf": 90, 
              "one_hot_max_size": 50, 
#               "task_type": "GPU", 
              'colsample_bylevel': 0.61, 
              'thread_count': -1
#               "devices": "0",
             }
result = np.zeros((5, len(test_cp)))
offline_score1 = []
oof = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
for k, (train_idx, test_idx) in enumerate(skf.split(train_cp[[c for c in train_cp.columns if c not in ["id", "flag"]]], st)):
    dtrainx, dtrainy = train_cp.loc[train_idx, [c for c in train_cp.columns if c not in ["id", "flag"]]], train_user.flag[train_idx]
    dvalidx, dvalidy = train_cp.loc[test_idx, [c for c in train_cp.columns if c not in ["id", "flag"]]], train_user.flag[test_idx]
    cat_train = Pool(dtrainx.values, label=dtrainy.values, cat_features=cat_list)
    cat_valid = Pool(dvalidx.values, label=dvalidy.values, cat_features=cat_list)
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(cat_train, eval_set=cat_valid, use_best_model=True, early_stopping_rounds=200, verbose_eval=200)
    offline_score1.append(cat_model.best_score_['validation']['AUC'])
    oof.append(cat_model.predict_proba(dvalidx)[:, 1])
    result[k, :] = cat_model.predict_proba(test_cp[[c for c in test_cp.columns if c not in ["id", "flag"]]].values)[:, 1]
print("-" * 50)
print(offline_score1)
print("mean {} std {}".format(np.mean(offline_score1).round(6), np.std(offline_score1).round(6)))
np.save("./b_cat_result", result.mean(axis=0))

# xgboost(由于xgb无法直接处理类别特征，因此对众数特征做特殊处理)
all_user = train_cp.append(test_cp).reset_index(drop=True)
all_user['id'] = train_user.id.append(test_user.id).reset_index(drop=True)
all_trd = train_trd[['id', 'cny_trx_amt']].append(test_trd[['id', 'cny_trx_amt']]).append(test_trd[['id', 'cny_trx_amt']]).reset_index(drop=True)
all_trd = all_trd.merge(all_user.reset_index(drop=True), on='id', how='left')
for c in gain_cat_feats:
    train_cp[c+"_cny_median"] = train_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"_cny_median"] = test_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"_cny_sum"] = train_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"_cny_sum"] = test_cp[c].astype(object).map(all_trd.groupby(c)['cny_trx_amt'].sum()).astype(float)
    train_cp[c+"pos_cny_median"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"pos_cny_median"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"pos_cny_sum"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"pos_cny_sum"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt > 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    train_cp[c+"neg_cny_median"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    test_cp[c+"neg_cny_median"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].median()).astype(float)
    train_cp[c+"neg_cny_sum"] = train_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    test_cp[c+"neg_cny_sum"] = test_cp[c].astype(object).map(all_trd.loc[all_trd.cny_trx_amt < 0].groupby(c)['cny_trx_amt'].sum()).astype(float)
    del train_cp[c], test_cp[c]
xgb_params = {
        'booster': 'gbtree',
        "objective": "binary:logistic",
        'eval_metric': 'auc',
        "verbosity": 0,
        # "tree_method": "gpu_hist",
        "max_depth": 9,
        "seed": 999,
        'eta': 0.01,
        'num_leaves': 300,
        'colsample_bytree': 0.61,
        'subsample': 0.87,
        "min_child_weight": 1.5,
    }
result = np.zeros((5, len(test_cp)))
offline_score1 = []
oof = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
for k, (train_idx, test_idx) in enumerate(skf.split(train_cp[[c for c in train_cp.columns if c not in ["id", "flag"]]], st)):
    xtrain = xgb.DMatrix(train_cp.values[train_idx], label=train_user.flag[train_idx])
    xtest = xgb.DMatrix(train_cp.values[test_idx], label=train_user.flag[test_idx])
    eval_result = {}
    xgb_model = xgb.train(xgb_params, 
                          xtrain, 
                          num_boost_round=1000, 
                          evals=[(xtrain, "train"), (xtest, 'eval')], 
                          evals_result=eval_result,
                          early_stopping_rounds=200, 
                          verbose_eval=100)
    offline_score1.append(eval_result['eval']['auc'][-1])
    oof.append(xgb_model.predict(xtest))
    result[k, :] = xgb_model.predict(xgb.DMatrix(test_cp[[c for c in test_cp.columns if c not in ["id", "flag"]]].values))
print("-" * 50)
print(offline_score1)
print("mean {} std {}".format(np.mean(offline_score1).round(6), np.std(offline_score1).round(6)))
np.save("./b_xgb_result.npy", result.mean(0))

# 简单平均
final_result = (np.load("./b_lgb_result.npy") + np.load("./b_cat_result.npy") + np.load("./b_xgb_result.npy")) / 3
submit = pd.DataFrame({'id': test_user.id, 'score': final_result})
submit.to_csv("./{}{}{}.txt".format(datetime.now().day, datetime.now().hour, datetime.now().minute), header=None, index=None, sep='\t')