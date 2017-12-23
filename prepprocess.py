import pandas as pd 
import numpy as np 
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Preprocess script of Deep_Brain .')
parser.add_argument('-input_file')
parser.add_argument('-output_dir')
arg = parser.parse_args()

alpha = 3

def get_no_means_columns(data):
    
    no_means_columns = [ column for column in data.columns 
                         if len(set(data[column].values)==1) or data[column].isna().sum() > len(data[column])*0.5]

    return no_means_columns

def get_no_means_rows(data, mean_std):

    row_abnormal_count = defaultdict(lambda:0)
    
    no_means_rows = [ row for row in data.index 
                       if data.iloc[row].isna().sum() > len(data.iloc[row])*0.5 ]
    
    for column in data.columns:
        t = data[column]
        
        if t.dtype == 'object':
            continue
        
        mean, std = mean_std(column)

        for ind, (flag1, flag2) in enumerate(t < (mean - std*alpha), t > (mean + std*alpha)):
            if flag1 or flag2:
                row_abnormal_count[ind] += 1

        abnormal_value_bound = len(data.columns) * 0.1
        for k in row_abnormal_count:
            if row_abnormal_count[k] > abnormal_value_bound and k not in no_means_rows:
                no_means_rows.append(IDs[k])

    return no_means_rows

def get_obj_columns(data):
    obj_replace_rule = {}
    
    obj_columns = [ column for column in data.columns
                    if data[column].dtype == 'object']
    
    for column in obj_columns:
        items = set(data[column])
        #item_count = len(items)    测试集中是否可能存在训练集不存在的值（顺延）
        obj_replace_rule[column] = {item:ind for ind, item in enumerate(items)}

    return obj_columns, obj_replace_rule

def get_mean_std(data):
    
    mean_std = {}
    
    for column in data.columns:
        t = data[column]
        if t.dtype == 'object':
            continue
        mean_std(column) = (t.mean(), t.std())

    return mean_std

def get_min_max(data, mean_std):
    
    min_max = {}
    
    for column in data.columns:
        t = data[column]
        mean, std = mean_std(column)
        if t.dtype == 'object':
            continue
        values = t[t > (mean + std*alpha)] = mean + std*alpha
        values = t[t < (mean + std*alpha)] = mean - std*alpha
        min_max(column) = (values.min(), values.max())
    
    return min_max

# def data_filling(data):

#     return data.fillna(data.mean())


# def data_normalize(data, min_max):
    
#     for column in data.columns:
#         min_, max_ = min_max[column]
#         rg = max_ - min_
#         data[column] = data[column] / rg

#     return data

def data_normalize(series, min_, max_):
    rg = max_ - min_
    series = float(series - min_) / rg

def save_data(data, output_dir):
    save_path = output_dir + '/pretty_data.xlsx'
    data.to_excel(save_path)

def main():

    data = pd.read_excel(arg.input_file)
    data_bak = data
    data = data.iloc(:, :-1)
    mean_std = get_mean_std(data)
    min_max = get_min_max(data, mean_std)
    no_means_columns = get_no_means_columns(data)
    no_means_rows = get_no_means_rows(data, mean_std)
    obj_columns, obj_replace_rule = get_obj_columns(data)

    frame = pd.DataFrame()
    for column in data.columns:
        if column in no_means_columns:
            continue
        series = data[column]
        if column in obj_columns:
            replace_rule = obj_replace_rule(column)
            min_, max_ = min_max(column)
            series = pd.Series([replace_rule[d] for d in series], dtype = 'float64')
        series = series.fillna(series.mean())
        series = data_normalize(series)
        frame[column] = series

    frame['Y'] = data_bak.iloc[:, -1:]

    frame = frame.drop(no_means_rows)

    return frame













