#!/usr/bin/env python
# coding: utf-8

# ### Feature Engineering

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import requests
import json
import time
from tqdm import tqdm
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
from sklearn.cluster import KMeans


# In[67]:


df = pd.read_csv('train.csv')
df.head()


# ## This section handles cleaning and imputing categorical features.

# In[68]:


def refine_general_types(df):
    if "description" not in df.columns:
        df['description'] = ""

    def reclassify_row(row):
        if row['property_type_simplified'] != 'כללי':
            return row['property_type_simplified']

        desc = str(row.get('description', '')).lower()

        if re.search(r'\b(פנטהאוז|גג)\b', desc):
            return 'דירת גג'
        elif re.search(r'\b(סטודיו|לופט)\b', desc):
            return 'סטודיו'
        elif re.search(r'\bיחידת דיור\b', desc):
            return 'דירה'
        elif re.search(r'\bדירה\b', desc):
            return 'דירה'
        elif re.search(r'\b(וילה|בית פרטי|קוטג)\b', desc):
            return 'פרטי'
        else:
            return 'כללי'

    df['property_type_simplified'] = df.apply(reclassify_row, axis=1)
    return df

# In[80]:


def simplify_property_type(df):
    
    """
    Simplifies the 'property_type' column into six main categories:['דירה', 'דירת גג', 'דופלקס', 'דירת גן', 'פרטי', 'סטודיו'].

    - Rows clearly referring to non-apartment properties (e.g., parking or storage units)
      are identified and removed based on specific keywords.
    - Missing or ambiguous property types are initially labeled as 'כללי'.
    - Entries labeled as 'כללי' are further refined based on keywords found in the 'description' field.
    - The simplified result is assigned back to the 'property_type' column.

    Returns:
        pd.DataFrame: Cleaned dataset with only apartment-type properties.
    """
    def map_type(value):
        if pd.isna(value):
            return 'כללי'
        value = str(value).strip().lower()

        if re.search(r'(גג|פנטהאוז)', value):
            return 'דירת גג'
        elif re.search(r'דופלקס', value):
            return 'דופלקס'
        elif re.search(r'גן', value):
            return 'דירת גן'
        elif re.search(r'(פרטי|דו משפחתי|קוטג)', value):
            return 'פרטי'
        elif re.search(r'(סטודיו|לופט)', value):
            return 'סטודיו'
        elif re.search(r'(דירה|יחידת דיור|דירה להשכרה|החלפת דירות|סאבלט|מרתף|פרטר)', value):
            return 'דירה'
        
        # Now remove only if it's explicitly just a parking/storage property
        elif re.search(r'^(חניה|מחסן)(\s|/|$)', value):
            return 'remove'
        
        else:
            return 'כללי'

    df['property_type_simplified'] = df['property_type'].apply(map_type)
    
    # Save rows to be removed
    removed_df = df[df['property_type_simplified'] == 'remove'].copy()
    # Remove rows that were marked for removal
    cleaned_df = df[df['property_type_simplified'] != 'remove'].copy()
    # Refine 'כללי' values using description
    cleaned_df = refine_general_types(cleaned_df)
    cleaned_df['property_type'] = cleaned_df['property_type_simplified']
    cleaned_df.drop(columns='property_type_simplified', inplace=True)

    return cleaned_df


# In[79]:


def remove_ad_rows(df):
    """
    Removes rows that appear to be unrelated ads based on the description.
    
    A row is removed if:
    - It contains exact ad-related keywords (e.g., מכירה, מצלמה, דרושים)
    - It does NOT contain any form of residential keywords (e.g., דירה, להשכרה, סטודיו)
    
    Returns:
        cleaned_df: DataFrame without the suspected ads
    """
    if "description" not in df.columns:
        df['description'] = ""
        
    # Ad keywords (must appear exactly)
    ad_keywords = r'(?:מכירה|למכירה|מצלמה|מצלמת|משרד|חנות|דרושים|חניה|מחסן|משרה|חדשות)'

    # Residential keywords (can appear in any form)
    residential_keywords = (
        r'(?:דיר[י|ה|ת]?|החלפת דירות|דייר|יחידת דיור|דיור|להשכרה|לופט|לזוג|סאבלט|מרתף|פרטר|מקלט|גג|פנטהאוז|חדר|דופלקס|'
        r'דירת גן|דו משפחתי|פרטי|קוטג\'|סטודיו|השכרה)'
    )

        # Ensure description is string and clean punctuation
    df.loc[:, 'description'] = df['description'].fillna('').astype(str)
    df.loc[:, 'description'] = df['description'].str.replace(r'[^\w\s\u0590-\u05FF]', ' ', regex=True)

    # Flag rows matching ad keywords AND not matching residential ones
    is_ad = df['description'].str.contains(ad_keywords, case=False, na=False) & \
            ~df['description'].str.contains(residential_keywords, case=False, na=False)

    ads_df = df[is_ad].copy()
    cleaned_df = df[~is_ad].copy()

    return cleaned_df


# In[81]:


def fill_neighborhood_from_street(df):
    """
    Fills missing 'neighborhood' values using the most common
    neighborhood found for the same street name (extracted from 'address').
    It builds the mapping ONLY from the input df to avoid data leakage.

    Parameters:
        df (DataFrame): The dataset to clean.

    Returns:
        df (DataFrame): The updated dataset with 'neighborhood' filled where possible.
    """
    if df['neighborhood'].isna().sum() == 0:
        return df  # nothing to fill

    def extract_street_name(address):
        if pd.isna(address):
            return None
        match = re.search(r'^[^\d,]+', address.strip())
        return match.group(0).strip() if match else None

    # Compute street name for all rows
    df['street_name'] = df['address'].apply(extract_street_name)

    # Build mapping only from rows that already have a known neighborhood
    known = df[df['neighborhood'].notna()].copy()
    known['street_name'] = known['address'].apply(extract_street_name)

    # Create a mapping: street → most frequent neighborhood
    street_to_neighborhood = (
        known.groupby('street_name')['neighborhood']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    # Fill in the missing neighborhoods using the mapping
    mask_missing = df['neighborhood'].isna()
    df.loc[mask_missing, 'neighborhood'] = df.loc[mask_missing, 'street_name'].map(street_to_neighborhood)

    # Clean up temporary column
    df.drop(columns='street_name', inplace=True)

    return df


# # This section handles cleaning and imputing numeric features.
# Missing values are filled using context-aware methods (like similar-area median),
# or dropped if no reliable estimation is possible.

# In[82]:


def fill_room_num_by_area(df):
    """
    Fills missing or zero values in the 'room_num' column using property type and area information.

    Logic:
    - If the property is labeled 'סטודיו' and has no room number → sets room_num to 1.
    - For other missing/zero values:
        - Attempts to fill using the median 'room_num' of apartments with ±5 sqm area.
        - If no similar apartments are found, the row is left unchanged.
    - Reclassifies:
        - room_num = 1 → 'סטודיו'
        - 'סטודיו' with more than 1.5 rooms → 'דירה'

    Returns:
        pd.DataFrame: Updated DataFrame with inferred room numbers and adjusted property types.
"""

    
    #Directly fill room_num = 1 for studio listings
    mask_studio_fix = ((df['room_num'].isna()) | (df['room_num'] == 0)) & (df['property_type'] == 'סטודיו')
    df.loc[mask_studio_fix, 'room_num'] = 1
    
    #Fill other missing/zero values using area-based median
    rows_to_drop = []
    missing_indexes = df[(df['room_num'].isna()) | (df['room_num'] == 0)].index
    updates = {}
    for idx in missing_indexes:
        area_value = df.loc[idx, 'area']
        
        if pd.notna(area_value):
            # Find apartments with similar area (±5) and non-null room_num
            similar = df[
                (df['area'].between(area_value - 5, area_value + 5)) &
                (df['room_num'].notna()) & (df['room_num'] > 0)
            ]
            if not similar.empty:
                updates[idx] = similar['room_num'].median()
            else:
                # if similar is empty, delete row
                rows_to_drop.append(idx)
        else:
            # if area is also missing, delete row
            rows_to_drop.append(idx)
            
    for idx, value in updates.items():
        df.at[idx, 'room_num'] = value
    
    # Reclassify room_num = 1 as 'סטודיו'
    df.loc[df['room_num'] == 1, 'property_type'] = 'סטודיו'
    # Reclassify incorrect 'סטודיו' with too many rooms as 'דירה'
    df.loc[(df['property_type'] == 'סטודיו') & (df['room_num'] > 1.5), 'property_type'] = 'דירה'
    return df



# #### Handling float values in the 'floor' column
# 
# We round floor values **up** just in case such cases appear in the test set. 
# 
# This choice is based on the idea that real estate listings often treat partial floors (like 2.5) as being on the higher level — for example, 2.5 is usually closer to the 3rd floor than the 2nd. Rounding up avoids underestimating the apartment's floor and keeps the data consistent. 
# 
# 

# In[83]:


def _safe_int(text):
    """
    Safely converts a text value to integer.
    If conversion fails, returns NaN instead of raising an error.
    """

    try:
        return int(text)
    except:
        return np.nan

def fix_floor_values_that_exceed_total(df):
    """
    Identifies and corrects cases where the 'floor' value appears to be a merged integer
    representing both the floor and total number of floors (e.g., 810 → floor 8, total_floors 10).
    Updates 'floor' accordingly where a valid split is detected.
    
    """
    mask = (df['floor'].notna() & df['total_floors'].notna() & (df['floor'] > df['total_floors']))
    df_fix = df.loc[mask].copy()
    df_fix['floor_str'] = df_fix['floor'].astype(str)

    
    max_len_val = df_fix['floor_str'].str.len().max()
    if pd.notna(max_len_val):
        max_len = int(max_len_val)
    else:
        max_len = 0 

    for split_at in range(1, max_len):
        floor_part = pd.to_numeric(df_fix['floor_str'].str[:split_at], errors='coerce')
        total_part = pd.to_numeric(df_fix['floor_str'].str[split_at:], errors='coerce')

        match_mask = total_part == df_fix['total_floors']
        df.loc[df_fix[match_mask].index, 'floor'] = floor_part[match_mask]

    return df

def clean_floor_column(df):
    """
    Cleans and standardizes the 'floor' column and associated 'total_floors'.

    - Parses strings like '2 מתוך 3' or 'קרקע' into separate numeric fields.
    - If 'total_floors' is missing or incorrect, fills/corrects it based on parsed values.
    - Identifies and fixes cases where floor and total_floors are merged into one number (e.g., 810 → 8).
    - Rounds up any float values in 'floor' to the nearest integer.
    - Ensures both 'floor' and 'total_floors' are clean and consistent.
    """

    def extract_floor_and_total(value):
        if pd.isna(value):
            return (np.nan, np.nan)
        if isinstance(value, (int, float)):
            return (value, np.nan)

        value = str(value).strip()
        parts = value.split("מתוך")

        if len(parts) == 2:
            floor_part = parts[0].strip()
            total_part = parts[1].strip()
            floor = 0 if floor_part == 'קרקע' else _safe_int(floor_part)
            total = 0 if total_part == 'קרקע' else _safe_int(total_part)
            return (floor, total)

        floor = 0 if value == 'קרקע' else _safe_int(value)
        return (floor, np.nan)

    # 1. Parse 'floor' into numeric floor + total_floors
    parsed = df['floor'].apply(extract_floor_and_total)
    df[['parsed_floor', 'parsed_total_floors']] = pd.DataFrame(parsed.tolist(), index=df.index)

    # 2. Apply parsed floor
    df['floor'] = df['parsed_floor']
    df.drop(columns='parsed_floor', inplace=True)

    # 3. Fill or correct total_floors
    fill_mask = df['total_floors'].isna() & df['parsed_total_floors'].notna()
    df.loc[fill_mask, 'total_floors'] = df.loc[fill_mask, 'parsed_total_floors']

    correct_mask = (
        df['parsed_total_floors'].notna() & 
        (df['total_floors'] != df['parsed_total_floors'])
    )
    df.loc[correct_mask, 'total_floors'] = df.loc[correct_mask, 'parsed_total_floors']
    df.drop(columns='parsed_total_floors', inplace=True)

    # 4. Fix values like 810 → 8
    df = fix_floor_values_that_exceed_total(df)

    # 5. Round up float floor values and convert to int
    df['floor'] = df['floor'].apply(lambda x: math.ceil(x) if pd.notna(x) else x)
    df['floor'] = df['floor'].apply(lambda x: int(x) if pd.notna(x) else x)

    # Optional: drop rows where floor is still NaN
    # df = df[df['floor'].notna()]

    return df


# In[84]:


def fill_area_by_room_num(df):
    """
    Fills missing or zero 'area' values, as well as area values < 20,
    using the median area of apartments with the same 'room_num',
    only if 'room_num' > 1.
    """
    # Replace zeros with NaN for consistency
    df['area'] = df['area'].replace(0, np.nan)
    
    # Fill based on median area per room_num
    median_area = df[df['area'] >= 20].groupby('room_num')['area'].median()

    # Identify rows needing fill
    mask = ((df['area'].isna()) | (df['area'] < 20)) & (df['room_num'] > 1.5)

    for idx in df[mask].index:
        room_val = df.loc[idx, 'room_num']
        if room_val in median_area:
            df.at[idx, 'area'] = median_area[room_val]

    return df


# In[85]:


def fix_garden_area(df):
    """
    Ensures 'garden_area' is numeric and fills missing values with 0,
    assuming that missing means no garden.
    """
    df['garden_area'] = pd.to_numeric(df['garden_area'], errors='coerce').fillna(0)
    return df


# #### Days to enter
# We assume that -1, 0, and missing values all indicate immediate availability. 
# To ensure consistency, we set all these values to 0.

# In[86]:


def fix_days_to_enter(df):
    """
    Replaces missing, -1, or 0 values in 'days_to_enter' with 0,
        assuming all mean immediate availability.
    """
    print(df.columns.tolist())
    
    if 'days_to_enter' in df.columns:
        df['days_to_enter'] = df['days_to_enter'].replace(-1, 0).fillna(0)
    else:
        print("העמודה 'days_to_enter' לא קיימת")

    #df['days_to_enter'] = df['days_to_enter'].replace(-1, 0).fillna(0)
    return df


# In[87]:


def fix_num_of_payments(df):
    """
    Fills missing and zero values in 'num_of_payments'
    using the mode per room_num. Falls back to global mode.
    """
    df['num_of_payments'] = df['num_of_payments'].replace(0, np.nan)

    global_mode = df['num_of_payments'].mode().iloc[0] if not df['num_of_payments'].mode().empty else 1

    df['num_of_payments'] = df.groupby('room_num')['num_of_payments'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else global_mode))

    return df


# In[88]:


def fill_monthly_arnona(df):
    """
    Fills missing 'monthly_arnona' values using:
    0. Removes outliers (e.g., Arnona > 2000)
    1. Descriptions explicitly stating Arnona is included
    2. Regex-based fallback for softer phrasing
    3. Median by (neighborhood, area_bucket)
    4. Median by (neighborhood, room_num)
    5. KNN Imputation for remaining missing values

    Assumes:
    - 0 means Arnona is included and should not be replaced.
    - 'neighborhood', 'area', and 'room_num' are already cleaned.
    - No leakage from target columns (like price).
    """
    
    if "description" not in df.columns:
        df['description'] = ""
        
    # ---- Step 0: Handle outliers ----
    # Treat unusually high Arnona values as missing (e.g., > 2000 NIS)
    df.loc[df['monthly_arnona'] > 2000, 'monthly_arnona'] = np.nan

    # ---- Step 1: Exact phrase match in description ----
    def arnona_included_safe(text):
        if pd.isna(text):
            return False
        text = str(text).lower().replace('\u200f', '').strip()
        return any(kw in text for kw in ['כולל ארנונה', 'כולל הכל', 'הארנונה כלולה', 'בעל הבית משלם ארנונה', 'כולל חשבונות'])

    def has_negative_phrase(text):
        if pd.isna(text):
            return False
        text = str(text).lower().replace('\u200f', '').strip()
        return 'לא כולל ארנונה' in text or 'בלי ארנונה' in text

    mask_from_description = (df['monthly_arnona'].isna() & df['description'].apply(arnona_included_safe) &
        ~df['description'].apply(has_negative_phrase))
    df.loc[mask_from_description, 'monthly_arnona'] = 0

    # ---- Step 2: Regex fallback for looser patterns ----
    arnona_regex = re.compile(r"(כולל.*ארנונה|ארנונה.*כולל|כלול.*ארנונה)", flags=re.IGNORECASE)

    def regex_arnona_included(text):
        if pd.isna(text):
            return False
        return bool(arnona_regex.search(str(text)))

    regex_mask = df['monthly_arnona'].isna() & df['description'].apply(regex_arnona_included)
    df.loc[regex_mask, 'monthly_arnona'] = 0

    # ---- Step 3: Median by (neighborhood, area_bucket) ----
    df['area_bucket'] = (df['area'] // 5) * 5
    median_by_area = (df[df['monthly_arnona'].notna()].groupby(['neighborhood', 'area_bucket'])['monthly_arnona'].median())

    def fill_by_area(row):
        if pd.notna(row['monthly_arnona']) or row['monthly_arnona'] == 0:
            return row['monthly_arnona']
        return median_by_area.get((row['neighborhood'], row['area_bucket']), np.nan)

    df['monthly_arnona'] = df.apply(fill_by_area, axis=1)

    # ---- Step 4: Median by (neighborhood, room_num) ----
    median_by_room = (df[df['monthly_arnona'].notna()].groupby(['neighborhood', 'room_num'])['monthly_arnona'].median())

    def fill_by_room(row):
        if pd.notna(row['monthly_arnona']) or row['monthly_arnona'] == 0:
            return row['monthly_arnona']
        return median_by_room.get((row['neighborhood'], row['room_num']), np.nan)

    df['monthly_arnona'] = df.apply(fill_by_room, axis=1)

    # ---- Step 5: KNN Imputation for remaining missing values ----
    # Encode neighborhood safely
    df['neighborhood_encoded'] = df['neighborhood'].astype('category').cat.codes
    # One-hot encode property_type
    property_dummies = pd.get_dummies(df['property_type'], prefix='property')
    df['elevator'] = df['elevator'].fillna(0)
    
    # Columns to use for KNN 
    knn_df = pd.concat([df[['area', 'room_num', 'neighborhood_encoded','elevator', 'monthly_arnona']], property_dummies], axis=1)
    features_to_scale = ['area', 'room_num', 'neighborhood_encoded']
    scaler = StandardScaler()
    knn_df[features_to_scale] = scaler.fit_transform(knn_df[features_to_scale])
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(knn_df)
    df['monthly_arnona'] = imputed[:, knn_df.columns.get_loc('monthly_arnona')]

    # ---- Cleanup ----
    df.drop(columns=['area_bucket', 'neighborhood_encoded'], inplace=True)

    return df


# In[89]:


def fix_building_tax(df):
    """
    Fills missing values in 'building_tax' with 0.
    
    We assume that if the value is missing, it means either:
    - There is no building maintenance fee to pay (e.g., in old/private buildings), or
    - The fee is already included in the rent price.
    
    We apply the same assumption to existing 0 values, so there is no need to modify them.
    """

    df['building_tax'] = df['building_tax'].fillna(0)
    
    return df


# In[90]:


def clean_total_floors(df):
    
    """
    Cleans the 'total_floors' column:

    - If 'floor' is 0 (ground floor) and 'total_floors' is missing → fills total_floors with 0.
      This assumes such properties are private houses or low-rise buildings.

    - Then, removes any remaining rows where 'total_floors' is still missing,
      regardless of whether 'floor' is present or not.

    Note:
    This step is applied after parsing combined 'floor' strings like '2 מתוך 3',
    where 'total_floors' might have already been extracted.
    """

    
    # Fill total_floors = 0 when floor is 0
    mask = (df['total_floors'].isna()) & (df['floor'] == 0)
    df.loc[mask, 'total_floors'] = 0

    # Drop rows where total_floors is missing and there's no way to fill it
    rows_to_drop = df[(df['total_floors'].isna())].index
    df = df.drop(index=rows_to_drop)

    return df


# In[91]:


def clean_price_column(df, min_price=1500, max_price=60000):
    """
    Cleans the 'price' column:
    - Removes extreme values (price < min_price or > max_price)
    - Removes rows with missing price
    - Drops rows where 'monthly_arnona' or 'building_tax' > price
    """

    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    df = df[df['price'].notna()]
    df = df[(df['monthly_arnona'].isna()) | (df['monthly_arnona'] <= df['price'])]
    df = df[(df['building_tax'].isna()) | (df['building_tax'] <= df['price'])]
    
    return df


# In[92]:


def clean_distance_from_center_column(df, dataset_type="train"):
    """
    Cleans the 'distance_from_center' column:
    - Converts values below 15 (assumed to be in km) to meters
    - Fills missing or zero values using Google Maps API (if API key is provided)
    - Falls back to neighborhood median saved from train set
    - Saves medians during train and loads them during test
    - Re-checks values over 12,000 meters and replaces them
    """
    
    api_key = None  # Google Maps API key (set to None to disable)
    center_address = "Dizengoff Square, Tel Aviv"

    def get_distance_to_center(source_address):
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': api_key,
            'X-Goog-FieldMask': 'routes.distanceMeters'
        }
        body = {
            "origin": {"address": source_address},
            "destination": {"address": center_address},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE"
        }

        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            try:
                return response.json()['routes'][0]['distanceMeters']
            except:
                return None
        return None

    # Step 1: Convert to numeric
    df['distance_from_center'] = pd.to_numeric(df['distance_from_center'], errors='coerce')

    # Step 2: Convert small values (assumed to be in km) to meters
    df['distance_from_center'] = df['distance_from_center'].apply(
        lambda x: int(x * 1000) if pd.notna(x) and x < 15 else x)

    # Step 3: Prepare neighborhood medians
    if dataset_type == "train":
        # Will compute and save at the end
        neighborhood_medians = {}
    else:
        # Load medians from file
        with open("neighborhood_distance_medians.pkl", "rb") as f:
            neighborhood_medians = pickle.load(f)

    # Step 4: Fill missing or 0 values
    missing_mask = (df['distance_from_center'].isna()) | (df['distance_from_center'] == 0)

    for idx in tqdm(df[missing_mask].index, desc="Filling missing distances"):
        address = df.loc[idx, 'address']
        neighborhood = df.loc[idx, 'neighborhood']
        full_address = None

        if pd.notna(address) and pd.notna(neighborhood):
            full_address = f"{address}, {neighborhood}, Tel Aviv"
        elif pd.notna(neighborhood):
            full_address = f"{neighborhood}, Tel Aviv"
        elif pd.notna(address):
            full_address = f"{address}, Tel Aviv"

        distance = None
        if api_key and full_address:
            distance = get_distance_to_center(full_address)

        if distance:
            df.at[idx, 'distance_from_center'] = int(distance)
        elif pd.notna(neighborhood):
            median_val = neighborhood_medians.get(neighborhood)
            if pd.notna(median_val):
                df.at[idx, 'distance_from_center'] = int(median_val)

    # Step 5: Recheck values > 12,000
    outlier_mask = df['distance_from_center'] > 12000

    for idx in tqdm(df[outlier_mask].index, desc="Rechecking large distances"):
        address = df.loc[idx, 'address']
        neighborhood = df.loc[idx, 'neighborhood']
        full_address = None

        if pd.notna(address) and pd.notna(neighborhood):
            full_address = f"{address}, {neighborhood}, Tel Aviv"
        elif pd.notna(neighborhood):
            full_address = f"{neighborhood}, Tel Aviv"
        elif pd.notna(address):
            full_address = f"{address}, Tel Aviv"

        distance = None
        if api_key and full_address:
            distance = get_distance_to_center(full_address)

        if distance and distance <= 12000:
            df.at[idx, 'distance_from_center'] = int(distance)
        elif pd.notna(neighborhood):
            median_val = neighborhood_medians.get(neighborhood)
            if pd.notna(median_val):
                df.at[idx, 'distance_from_center'] = int(median_val)

    # Step 6: Save neighborhood medians (only for train)
    if dataset_type == "train":
        medians = df.groupby('neighborhood')['distance_from_center'].median().to_dict()
        with open("neighborhood_distance_medians.pkl", "wb") as f:
            pickle.dump(medians, f)

    return df


# In[93]:


def remove_duplicate_samples(df):
    """
    Removes duplicate samples based on a defined set of feature columns.
    Keeps the first occurrence of each unique sample.
    """
    key_columns = [
        'property_type', 'neighborhood', 'address', 'room_num', 'floor', 'area',
        'garden_area', 'days_to_enter', 'num_of_payments', 'monthly_arnona',
        'building_tax', 'total_floors', 'has_parking', 'has_storage', 'elevator',
        'ac', 'handicap', 'has_bars', 'has_safe_room', 'has_balcony',
        'is_furnished', 'is_renovated', 'price'
    ]

    df = df.drop_duplicates(subset=key_columns, keep='first')

    return df


# In[94]:


def drop_rows_with_missing_critical_values(df):
    """
    Drops rows that have missing values in any of the critical columns required for modeling.
    These columns are considered essential for predictive performance and data integrity.
    (Used on the training set only, since it includes the target column 'price').

    Returns:
        cleaned_df: DataFrame with rows containing missing values in the critical columns removed.
    """

    critical_columns = [
        'property_type', 'neighborhood', 'room_num', 'floor', 'area', 
        'garden_area', 'days_to_enter', 'num_of_payments', 'monthly_arnona',
        'building_tax', 'total_floors', 'description', 'has_parking',
        'has_storage', 'elevator', 'ac', 'handicap', 'has_bars',
        'has_safe_room', 'has_balcony', 'is_furnished', 'is_renovated', 'price',
        'num_of_images', 'distance_from_center'
    ]
    
    cleaned_df = df.dropna(subset=critical_columns)
    
    return cleaned_df


def kfold_target_encoding(df, col, target, n_splits=5, global_smoothing=10):

    
    """
    K-Fold Target Encoding prevents leakage by encoding each row using out-of-fold data.
    This approach provides more reliable feature values for training models, especially in cross-validation settings.

    Saves the final target means for use in test-time encoding.
    """

    df = df.copy()
    global_mean = df[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = pd.Series(index=df.index, dtype=float)

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]

        means = train_fold.groupby(col)[target].mean()
        val_encoded = val_fold[col].map(means)
        val_encoded.fillna(global_mean, inplace=True)
        encoded.iloc[val_idx] = val_encoded

    # Save the final means from the full data (to use in test set later)
    final_means = df.groupby(col)[target].mean().to_dict()
    with open(f"{col}_means.pkl", "wb") as f:
        pickle.dump(final_means, f)

    return encoded

# In[95]:


def add_neighborhood_group_features(df, target_col='price', dataset_type='train', 
                                    n_clusters=10, bin_size=1000, quantile_bins=10):

    """
    Adds and encodes three neighborhood grouping strategies:
    1. KMeans clusters
    2. Fixed price binning
    3. Quantile binning

    Works for both 'train' and 'test' datasets.
    - In 'train': fits clusters, performs KFold target encoding, and saves mappings.
    - In 'test': loads mappings and applies them.

    Returns:
        df (pd.DataFrame): DataFrame with 3 new **encoded** columns.
    """
    df = df.copy()

    if dataset_type == 'train':
        # Group by neighborhood and calculate mean prices
        mean_prices = df.groupby('neighborhood')[target_col].mean().reset_index()
        mean_prices.columns = ['neighborhood', 'mean_price']

        # --- 1. KMeans Clustering ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        mean_prices['kmeans_cluster'] = kmeans.fit_predict(mean_prices[['mean_price']])
        map_kmeans = dict(zip(mean_prices['neighborhood'], mean_prices['kmeans_cluster']))
        df['neighborhood_group_kmeans'] = df['neighborhood'].map(map_kmeans)

        # --- 2. Fixed Binning ---
        mean_prices['bin_cluster'] = (mean_prices['mean_price'] // bin_size).astype(int)
        map_bins = dict(zip(mean_prices['neighborhood'], mean_prices['bin_cluster']))
        df['neighborhood_group_bins'] = df['neighborhood'].map(map_bins)

        # --- 3. Quantile Binning ---
        try:
            mean_prices['quantile_cluster'] = pd.qcut(mean_prices['mean_price'], q=quantile_bins, labels=False, duplicates='drop')
        except ValueError:
            mean_prices['quantile_cluster'] = pd.cut(mean_prices['mean_price'], bins=quantile_bins, labels=False)
        map_quantiles = dict(zip(mean_prices['neighborhood'], mean_prices['quantile_cluster']))
        df['neighborhood_group_quantiles'] = df['neighborhood'].map(map_quantiles)

        # --- Target encoding for grouped features ---
        df['neighborhood_group_kmeans_encoded'] = kfold_target_encoding(df, 'neighborhood_group_kmeans', target_col)
        df['neighborhood_group_bins_encoded'] = kfold_target_encoding(df, 'neighborhood_group_bins', target_col)
        df['neighborhood_group_quantiles_encoded'] = kfold_target_encoding(df, 'neighborhood_group_quantiles', target_col)

        # --- Save all mappings and means ---
        with open("kmeans_mapping.pkl", "wb") as f:
            pickle.dump(map_kmeans, f)
        with open("bins_mapping.pkl", "wb") as f:
            pickle.dump(map_bins, f)
        with open("quantiles_mapping.pkl", "wb") as f:
            pickle.dump(map_quantiles, f)

        with open("neighborhood_group_kmeans_means.pkl", "wb") as f:
            pickle.dump(df.groupby('neighborhood_group_kmeans')[target_col].mean().to_dict(), f)
        with open("neighborhood_group_bins_means.pkl", "wb") as f:
            pickle.dump(df.groupby('neighborhood_group_bins')[target_col].mean().to_dict(), f)
        with open("neighborhood_group_quantiles_means.pkl", "wb") as f:
            pickle.dump(df.groupby('neighborhood_group_quantiles')[target_col].mean().to_dict(), f)

    elif dataset_type == 'test':
        # --- Load mappings ---
        with open("kmeans_mapping.pkl", "rb") as f:
            map_kmeans = pickle.load(f)
        with open("bins_mapping.pkl", "rb") as f:
            map_bins = pickle.load(f)
        with open("quantiles_mapping.pkl", "rb") as f:
            map_quantiles = pickle.load(f)

        df['neighborhood_group_kmeans'] = df['neighborhood'].map(map_kmeans)
        df['neighborhood_group_bins'] = df['neighborhood'].map(map_bins)
        df['neighborhood_group_quantiles'] = df['neighborhood'].map(map_quantiles)

        # --- Load encoding means ---
        with open("neighborhood_group_kmeans_means.pkl", "rb") as f:
            kmeans_means = pickle.load(f)
        with open("neighborhood_group_bins_means.pkl", "rb") as f:
            bins_means = pickle.load(f)
        with open("neighborhood_group_quantiles_means.pkl", "rb") as f:
            quantiles_means = pickle.load(f)

        df['neighborhood_group_kmeans_encoded'] = df['neighborhood_group_kmeans'].map(kmeans_means).fillna(np.mean(list(kmeans_means.values())))
        df['neighborhood_group_bins_encoded'] = df['neighborhood_group_bins'].map(bins_means).fillna(np.mean(list(bins_means.values())))
        df['neighborhood_group_quantiles_encoded'] = df['neighborhood_group_quantiles'].map(quantiles_means).fillna(np.mean(list(quantiles_means.values())))

    # Drop intermediate cluster columns to avoid redundancy
    df.drop(columns=['neighborhood_group_kmeans', 'neighborhood_group_bins', 'neighborhood_group_quantiles'], errors='ignore', inplace=True)

    return df


# In[96]:


def add_distance_group_features(df, target_col='price', dataset_type=None,
                                 fixed_bins=[0, 2000, 4000, 6000, 10000, np.inf],
                                 n_clusters=5, quantile_bins=5):
    """
    Adds distance-based groupings to df:
    - Fixed binning
    - Quantile-based binning (consistent bins saved for test)
    - KMeans clustering

    Then applies KFold Target Encoding (only for train) and saves mappings.
    For 'test', uses saved bins/models to ensure consistent transformations.

    Drops raw grouping columns after encoding.
    """
    if dataset_type not in ['train', 'test']:
        raise ValueError("dataset_type must be either 'train' or 'test'")

    df = df.copy()

    # --- Fixed Binning (same for both train/test) ---
    df['distance_group_fixed'] = pd.cut(df['distance_from_center'], bins=fixed_bins, labels=False)

    # --- Quantile Binning (train: compute, test: load mapping) ---
    if dataset_type == 'train':
        try:
            df['distance_group_quantiles'] = pd.qcut(df['distance_from_center'], q=quantile_bins, labels=False, duplicates='drop')
        except:
            df['distance_group_quantiles'] = pd.cut(df['distance_from_center'], bins=quantile_bins, labels=False)
    else:
        with open("distance_group_quantiles_bins.pkl", "rb") as f:
            bins = pickle.load(f)
        df['distance_group_quantiles'] = pd.cut(df['distance_from_center'], bins=bins, labels=False)

    # --- KMeans Clustering ---
    if dataset_type == 'train':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['distance_group_kmeans'] = kmeans.fit_predict(df[['distance_from_center']])
        with open("kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
    else:
        with open("kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
        df['distance_group_kmeans'] = kmeans.predict(df[['distance_from_center']])

    if dataset_type == 'train':
        # Encode with target
        df['distance_group_fixed_encoded'] = kfold_target_encoding(df, 'distance_group_fixed', target_col)
        df['distance_group_quantiles_encoded'] = kfold_target_encoding(df, 'distance_group_quantiles', target_col)
        df['distance_group_kmeans_encoded'] = kfold_target_encoding(df, 'distance_group_kmeans', target_col)

        # Save means
        with open("distance_group_fixed_means.pkl", "wb") as f:
            pickle.dump(df.groupby('distance_group_fixed')[target_col].mean().to_dict(), f)
        with open("distance_group_quantiles_means.pkl", "wb") as f:
            pickle.dump(df.groupby('distance_group_quantiles')[target_col].mean().to_dict(), f)
        with open("distance_group_kmeans_means.pkl", "wb") as f:
            pickle.dump(df.groupby('distance_group_kmeans')[target_col].mean().to_dict(), f)

        # Save bins used for quantile
        qcut_bins = pd.qcut(df['distance_from_center'], q=quantile_bins, retbins=True, duplicates='drop')[1]
        with open("distance_group_quantiles_bins.pkl", "wb") as f:
            pickle.dump(qcut_bins, f)

    elif dataset_type == 'test':
        # Load means
        with open("distance_group_fixed_means.pkl", "rb") as f:
            fixed_means = pickle.load(f)
        with open("distance_group_quantiles_means.pkl", "rb") as f:
            quantile_means = pickle.load(f)
        with open("distance_group_kmeans_means.pkl", "rb") as f:
            kmeans_means = pickle.load(f)

        df['distance_group_fixed_encoded'] = df['distance_group_fixed'].map(fixed_means).fillna(np.mean(list(fixed_means.values())))
        df['distance_group_quantiles_encoded'] = df['distance_group_quantiles'].map(quantile_means).fillna(np.mean(list(quantile_means.values())))
        df['distance_group_kmeans_encoded'] = df['distance_group_kmeans'].map(kmeans_means).fillna(np.mean(list(kmeans_means.values())))

    # Drop raw grouping columns
    df.drop(columns=['distance_group_fixed', 'distance_group_quantiles', 'distance_group_kmeans'],
            errors='ignore', inplace=True)

    return df


# In[97]:





# In[98]:


def add_distance_bucket_feature(df):
    """
    Adds a new feature 'distance_bucket' by binning 'distance_from_center' into categorical ranges.
    This feature helps capture non-linear relationships between distance and price.
    
    The buckets are:
    - 0: 0–1500 meters
    - 1: 1500–3000 meters
    - 2: 3000–5000 meters
    - 3: 5000+ meters
    - -1: for missing values (NaNs)
    """
    df = df.copy()

    # Bin distance into labeled categories
    df['distance_bucket'] = pd.cut(df['distance_from_center'],bins=[0, 1500, 3000, 5000, np.inf],labels=[0, 1, 2, 3])

    # Convert to float to handle missing values, and fill NaNs with -1 to indicate "unknown"
    df['distance_bucket'] = df['distance_bucket'].astype(float).fillna(-1)

    return df


# In[100]:


def add_custom_features(df):
    """
    Adds domain-informed and interaction-based features to enrich the data.

    Includes:
    - Structural and spatial indicators (e.g., area_per_room, floor_ratio)
    - Feature interactions (e.g., area × is_furnished, floor × proximity)
    - Amenity-based quality scores
    - Distance bucketing using add_distance_bucket_feature()
    """


    # --- Structural & Spatial Features ---
    df['area_per_room'] = np.where(df['room_num'] > 0, df['area'] / df['room_num'], 0)  # layout efficiency
    df['floor_ratio'] = np.where(df['total_floors'] > 0, df['floor'] / df['total_floors'], 0)  # relative floor height
    df['is_top_floor'] = (df['floor'] == df['total_floors']).astype(int)
    df['has_parking_or_storage'] = ((df['has_parking'] == 1) | (df['has_storage'] == 1)).astype(int)
    df['is_penthouse_candidate'] = ((df['floor'] == df['total_floors']) & (df['total_floors'] > 10)).astype(int)
    df['is_high_floor_no_elevator'] = ((df['floor'] >= 4) & (df['elevator'] == 0)).astype(int)
    df['has_outdoor_space'] = ((df['has_balcony'] == 1) | (df['garden_area'] > 0)).astype(int)

    # --- Feature Interactions ---
    df['area_x_is_furnished'] = df['area'] * df['is_furnished']
    df['area_per_floor'] = df['area'] / (df['floor'] + 1)
    df['arnona_x_distance'] = df['monthly_arnona'] * df['distance_from_center']
    df['floor_x_proximity'] = df['floor_ratio'] * (1 / (df['distance_from_center'] + 1))
    df['garden_area_ratio'] = df['garden_area'] / (df['area'] + 1)
    df['area_x_balcony'] = df['area'] * df['has_balcony']
    df['arnona_per_sqm'] = df['monthly_arnona'] / (df['area'] + 1)

    if 'floor_ratio' in df.columns:
        df['floor_tax_interaction'] = df['floor_ratio'] * df['building_tax']

    if 'neighborhood_group_quantiles_encoded' in df.columns:
        df['group_quantiles_x_area'] = df['neighborhood_group_quantiles_encoded'] * df['area']

    if 'neighborhood_encoded' in df.columns and 'neighborhood_group_quantiles_encoded' in df.columns:
        df['neighborhood_vs_group'] = df['neighborhood_encoded'] - df['neighborhood_group_quantiles_encoded']

    # --- Quality Score & Amenities ---
    amenities = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap',
                 'has_bars', 'has_safe_room', 'is_furnished', 'is_renovated',
                 'has_outdoor_space']
    
    df['quality_score'] = df[amenities].sum(axis=1)
    df['has_5_plus_amenities'] = (df['quality_score'] >= 5).astype(int)

    if 'distance_from_center' in df.columns:
        df['quality_x_proximity'] = df['quality_score'] * (1 / (df['distance_from_center'] + 1))
        
    df = add_distance_bucket_feature(df)

    return df


# In[101]:


def remove_unused_features(df):
    """
    Removes features that were engineered but found to be unhelpful or redundant for the model.
    This step helps reduce dimensionality and avoids multicollinearity or noise.

    The list includes:
    - Raw features (e.g., has_parking, has_bars, room_num)
    - Intermediate engineered features (e.g., floor_tax_interaction, is_penthouse_candidate)
    - Encoded groups no longer needed
    """
    columns_to_remove = ['days_to_enter','is_top_floor','is_penthouse_candidate','is_high_floor_no_elevator'
                         'room_num', 'num_of_payments', 'total_floors', 'has_parking', 'has_storage','elevator',
                         'ac','handicap', 'has_bars','has_safe_room', 'has_balcony','is_furnished','is_renovated',
                         'num_of_images', 'neighborhood_group_bins_encoded','distance_group_fixed_encoded',
                         'distance_group_quantiles_encoded', 'distance_group_kmeans_encoded', 'has_outdoor_space',
                         'floor_ratio','area_per_room', 'floor_tax_interaction','has_5_plus_amenities','arnona_x_distance']   
    
    # Drop only columns that actually exist in the DataFrame
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    return df

