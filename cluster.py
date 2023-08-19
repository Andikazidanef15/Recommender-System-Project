import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import json

def aggregate_df(df):
    '''Agregate data based on user ID then get all the necessary information to do clustering
        Args:
        df (pd.DataFrame): Original Data stored in data folder
    '''
    # Read District Lat Long JSON File
    with open('coordinates.json') as json_file:
        district_lat_long = json.load(json_file)

    # Count menu type ordered by each user_id
    agg = df.groupby(['user_id', 'menu_type'], as_index = False)['menu_id'].count()
    agg = pd.pivot_table(agg, index = 'user_id', columns = 'menu_type', values = 'menu_id').fillna(0)

    # Count menu category ordered by each user_id
    agg_1 = df.groupby(['user_id', 'menu_category'], as_index = False)['menu_id'].count()
    agg_1 = pd.pivot_table(agg_1, index = 'user_id', columns = 'menu_category', values = 'menu_id').fillna(0)

    # Count menu category detailed ordered by each user_id
    agg_2 = df.groupby(['user_id', 'menu_category_detail'], as_index = False)['menu_id'].count()
    agg_2 = pd.pivot_table(agg_2, index = 'user_id', columns = 'menu_category_detail', values = 'menu_id').fillna(0)

    # Get Location of Each User ID and User Tier Level
    users = df.groupby(['user_id'])[['district','user_tier_level']].agg(lambda x: pd.Series.mode(x)[0])
    users['lat'] = users['district'].map(lambda district: district_lat_long[district][0])
    users['long'] = users['district'].map(lambda district: district_lat_long[district][1])
    users = pd.get_dummies(users, columns = ['user_tier_level'])

    # Combine into one DataFrame
    agg_all = pd.concat([agg, agg_1, agg_2, users.iloc[:, 1:]], axis = 1)

    # Scale Data
    scaler = MinMaxScaler()
    agg_all_scaled = pd.DataFrame(scaler.fit_transform(agg_all), columns = agg_all.columns, index = agg_all.index)

    return agg_all_scaled

def cluster(agg_all_scaled):
    '''Do customer segmentation using K-Means clustering for scaled aggregated data and retrieve closest users that represent each segment
        Args:
        df (pd.DataFrame): Original Data stored in data folder
    '''
    # Load K-Means Model
    with open("cluster_model/k_means.pkl", "rb") as f:
        km = pickle.load(f)

    # Predict
    agg_all_scaled['Label'] = km.predict(agg_all_scaled)

    # Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
    closest_pt_idx = []
    for iclust in range(km.n_clusters):
        # get all points assigned to each cluster:
        cluster_pts = agg_all_scaled[agg_all_scaled['Label'] == iclust]

        # get all indices of points assigned to this cluster:
        cluster_pts_indices = cluster_pts.index

        cluster_cen = km.cluster_centers_[iclust]
        min_idx = np.argmin([euclidean(agg_all_scaled.loc[idx].drop(['Label']), cluster_cen) for idx in cluster_pts_indices])

        # Testing:
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    
    return agg_all_scaled, closest_pt_idx
