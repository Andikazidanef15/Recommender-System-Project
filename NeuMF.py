import pytorch_lightning as pl
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.load_data import MenuTrainDataset, MenuTestDataset

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
        Args:
          num_factors (int): Number of embedding dimension
          nums_hiddens (list): List containing number of neurons in each layer
          num_users (int): Number of unique users
          num_items (int): Number of unique menus
          train_data (pd.DataFrame): Train Dataframe containing user and menu pair
          val_data (pd.DataFrame): Validation Dataframe containing user and menu pair
          all_menu_ids (list): List containing all menu_ids (train + test)
    """

    def __init__(self, num_factors, num_hiddens, num_users, num_items, num_negatives, 
                 batch_size, weight_decay, lr, train_data, val_data, 
                 all_menu_ids):
      super().__init__()
      self.save_hyperparameters()

      self.P = nn.Embedding(num_users, num_factors)
      self.Q = nn.Embedding(num_items, num_factors)
      self.U = nn.Embedding(num_users, num_factors)
      self.V = nn.Embedding(num_items, num_factors)
      self.mlp = nn.Sequential()
      for i in range(len(num_hiddens)):
        if i == 0:
          self.mlp.add_module(f'fc{i+1}', nn.Linear(num_factors * 2, num_hiddens[i]))
        else:
          self.mlp.add_module(f'fc{i+1}', nn.Linear(num_hiddens[i - 1], num_hiddens[i]))

      self.prediction_layer = nn.Linear(num_factors + num_hiddens[-1], 1, bias=False)
      self.train_data = train_data
      self.val_data = val_data
      self.all_menu_ids = all_menu_ids
      self.batch_size = batch_size
      self.weight_decay = weight_decay
      self.num_negatives = num_negatives
      self.lr = lr

    def forward(self, user_input, item_input):
      # Create Embedding for generic neural network version of matrix factorization (GMF)
      p_mf = self.P(user_input)
      q_mf = self.Q(item_input)

      # Elementwise Multiplication
      gmf = p_mf * q_mf

      # Create Embedding as input for MLP modules
      p_mlp = self.U(user_input)
      q_mlp = self.V(item_input)
      combined_mlp = torch.cat([p_mlp, q_mlp], dim=-1)
      mlp = nn.ReLU()(self.mlp(combined_mlp))

      con_res = torch.cat([gmf, mlp], dim = -1)
      pred = nn.Sigmoid()(self.prediction_layer(con_res))
      return pred

    def training_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)
      loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
      self.log('train_loss', loss)
      return loss

    def validation_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)
      loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
      self.log('val_loss', loss)
      return loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

    def train_dataloader(self):
      return DataLoader(MenuTrainDataset(self.train_data, self.all_menu_ids, self.num_negatives), batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
      return DataLoader(MenuTestDataset(self.val_data, self.all_menu_ids), batch_size=self.batch_size, num_workers=4)