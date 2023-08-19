# Import Package
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_user_encoding(df):
    '''Get Dictionary containing User ID and Menu ID nap to unique integers
    Args:
        df (pd.DataFrame): Original Dataframe (data.csv)
    '''
    # Create integer mapping for user id and menu id
    users_encode = [i for i in range(len(df['user_id'].unique()))]
    menus_encode = [i for i in range(len(df['menu_id'].unique()))]

    # Create dictionary for user id to int and menu id to int 
    dict_users_encode = dict(zip(df['user_id'].unique(), users_encode))
    dict_encode_users = dict(zip(users_encode, df['user_id'].unique()))
    dict_menus_encode = dict(zip(df['menu_id'].unique(), menus_encode))
    dict_encode_menus = dict(zip(menus_encode, df['menu_id'].unique()))

    return dict_users_encode, dict_encode_users, dict_menus_encode, dict_encode_menus

def encode_data(df):
    '''Encode User ID and Menu ID from Original Dataframe (df)
    Args:
        df (pd.DataFrame): Original Dataframe (data.csv)
    '''
    # Map User ID and Menu ID using Dictionaries from get_user_encoding
    dict_users_encode, _, dict_menus_encode, _ = get_user_encoding(df)
    df_model = df[['user_id','menu_id','trx_date']].copy()
    df_model['user_id'] = df_model['user_id'].map(dict_users_encode)
    df_model['menu_id'] = df_model['menu_id'].map(dict_menus_encode)

    return df_model

def split_data(df_model):
    # Split data into training and test data
    test = df_model.sort_values(by = ['user_id','trx_date'], ascending = False).drop_duplicates(subset = ['user_id'])
    train = df_model.drop(test.index, axis = 0)

    train = train[['user_id','menu_id']]
    test = test[['user_id','menu_id']]

    # Return Data
    return train, test

class MenuTrainDataset(Dataset):
    """PyTorch Dataset for Loading Training Data for Recommender System

    Args:
        data (pd.DataFrame): Dataframe containing the pair of user and menu
        all_menu_ids (list): List containing all menu ID

    """

    def __init__(self, data, all_menu_ids, num_negatives):
        self.users, self.items, self.labels = self.get_dataset(data, all_menu_ids, num_negatives)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, data, all_menu_ids, num_negatives):
        # Placeholders that will hold the training data
        users, items, labels = [], [], []

        # This is the set of items that each user has interaction with
        user_item_set = set(zip(data['user_id'], data['menu_id']))

        # 8:1 ratio of negative to positive samples
        num_negatives = num_negatives

        for (u, i) in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1) # items that the user has interacted with are positive
            for _ in range(num_negatives):
                # randomly select an item
                negative_item = np.random.choice(all_menu_ids)
                # check that the user has not interacted with this item
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_menu_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0) # items not interacted with are negative

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

class MenuTestDataset(Dataset):
    """PyTorch Dataset for Validation Data in Recommender Systen

    Args:
        data (pd.DataFrame): Dataframe containing the pair of user and menu
        all_menu_ids (list): List containing all menu ID

    """

    def __init__(self, data, all_menu_ids):
        self.users, self.items, self.labels = self.get_dataset(data, all_menu_ids)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, data, all_menu_ids):
        # Placeholders that will hold the training data
        users, items, labels = [], [], []

        # This is the set of items that each user has interaction with
        user_item_set = set(zip(data['user_id'], data['menu_id']))

        for (u, i) in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1) # items that the user has interacted with are positive

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)