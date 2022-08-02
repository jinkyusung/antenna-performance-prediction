import os
import re
import pandas as pd
import collections


def make_namegroup(df, info):
    """
    This function will return dictionary 
    that key is generalized name of each feature in {df}, value is each feature in {df}
    {info} is infomation csv file about feature of {df}.
    """
    namegroup = collections.defaultdict(list)
    info = info.set_index('Feature').T.to_dict('list')
    for feature in info:
        name = ' '.join(re.sub(r'[0-9]','n',*info[feature]).split())
        namegroup[name].append(feature)
    
    return namegroup


def make_dataframe(df, namegroup, feature_name):
    tmp = pd.DataFrame()
    for feature in namegroup[feature_name]:
        tmp[feature] = df[feature].copy()
    return tmp


def create_group_csvs(df, namegroup, save_route):
    for name in namegroup:
        tmp = pd.DataFrame()
        for feature in namegroup[name]:
            tmp[feature] = df[feature].copy()
        os.makedirs(f"{save_route}", exist_ok=True)
        tmp.to_csv(f"{save_route}/{name}.csv", index=False)