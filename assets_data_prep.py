#!/usr/bin/env python
# coding: utf-8

# ### Third Part


# In[19]:


import pandas as pd
import numpy as np
import re
import pickle

# Scikit-learn preprocessing and modeling
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer
)

# Feature engineering functions
from feature_engineering import (
    refine_general_types,
    simplify_property_type,
    remove_ad_rows,
    fill_neighborhood_from_street,
    fill_room_num_by_area,
    clean_floor_column,
    fill_area_by_room_num,
    fix_garden_area,
    fix_days_to_enter,
    fix_num_of_payments,
    fill_monthly_arnona,
    fix_building_tax,
    clean_total_floors,
    clean_price_column,
    clean_distance_from_center_column,
    remove_duplicate_samples,
    drop_rows_with_missing_critical_values,
    add_neighborhood_group_features,
    add_distance_group_features,
    kfold_target_encoding,
    add_custom_features,
    remove_unused_features
)


# In[20]:


def prepare_data(df, dataset_type):
    numeric_features = ['area', 'room_num', 'floor', 'distance_from_center', 'monthly_arnona', 'building_tax']
    for col in ['address', 'description', 'distance_from_center']:
        if col not in df.columns:
            df[col] = np.nan
    # 1. Feature Engineering
    df = simplify_property_type(df)
    df = remove_ad_rows(df)
    df = fill_neighborhood_from_street(df)
    df = fill_room_num_by_area(df)
    df = clean_floor_column(df)
    df = fill_area_by_room_num(df)
    df = fix_garden_area(df)
    df = fix_days_to_enter(df)
    df = fix_num_of_payments(df)
    df = fill_monthly_arnona(df)
    df = fix_building_tax(df)
    df = clean_total_floors(df)
    if dataset_type == "train":
        df = clean_price_column(df)
    df = clean_distance_from_center_column(df, dataset_type=dataset_type)
    if dataset_type == "train":
        df = remove_duplicate_samples(df)
        df = drop_rows_with_missing_critical_values(df)
    df = add_neighborhood_group_features(df, target_col='price', dataset_type=dataset_type)
    df = add_distance_group_features(df, target_col='price', dataset_type=dataset_type)
    if dataset_type == 'train':
        df['neighborhood_encoded'] = kfold_target_encoding(df, 'neighborhood', 'price')
        df['property_type_encoded'] = kfold_target_encoding(df, 'property_type', 'price')
        df = df.drop(columns=['neighborhood', 'property_type', 'address', 'description'])
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        # Save the fitted scaler to a file
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)        
        
    elif dataset_type == 'test':
        with open("neighborhood_means.pkl", "rb") as f:
            neighborhood_means = pickle.load(f)
        
        with open("property_type_means.pkl", "rb") as f:
            property_type_means = pickle.load(f)
            
        df['neighborhood_encoded'] = df['neighborhood'].map(neighborhood_means).fillna(np.mean(list(neighborhood_means.values())))
        df['property_type_encoded'] = df['property_type'].map(property_type_means).fillna(np.mean(list(property_type_means.values())))
        # Drop irrelevant/original columns (done for both train and test to align features)
        df = df.drop(columns=['neighborhood', 'property_type', 'address', 'description'], errors='ignore')
        
       # Load scaler from file
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df[numeric_features] = scaler.transform(df[numeric_features])
    df = add_custom_features(df)
    df = remove_unused_features(df)

    
    if dataset_type != 'train' and 'price' in df.columns:
        df = df.drop(columns=['price'])
        
    return df








