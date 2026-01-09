import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from config import WORK_DIR

# --- Dataset Class ---
class NeuMFDataset(Dataset):
    def __init__(self, df, members, songs, is_train=True):
        self.users = df['msno'].values
        self.items = df['song_id'].values
        self.is_train = is_train
        if is_train:
            self.targets = df['target'].values
        
        # Pre-fetch features from members/songs to avoid lookup overhead
        # Assuming df has 'msno' and 'song_id' which are indices
        # We need to map them if they aren't 0..N indices
        # But our preprocessing makes them 0..N Ints.
        
        self.members_city = members['city'].values
        self.members_gender = members['gender'].values
        self.members_reg = members['registered_via'].values
        
        # Ensure indices don't go out of bounds
        # self.users = np.clip(self.users, 0, len(members)-1)
        # self.items = np.clip(self.items, 0, len(songs)-1)
        
        self.songs_genre = songs['first_genre_id'].values
        # Add more if needed
        
        # Context from df
        self.tabs = df['source_system_tab'].values
        self.screens = df['source_screen_name'].values
        self.types = df['source_type'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        iid = self.items[idx]
        
        # User features
        city = self.members_city[uid]
        gender = self.members_gender[uid]
        reg = self.members_reg[uid]
        
        # Item features
        genre = self.songs_genre[iid]
        
        # Context
        tab = self.tabs[idx]
        screen = self.screens[idx]
        typ = self.types[idx]
        
        data = {
            'user': torch.tensor(uid, dtype=torch.long),
            'item': torch.tensor(iid, dtype=torch.long),
            'city': torch.tensor(city, dtype=torch.long),
            'gender': torch.tensor(gender, dtype=torch.long),
            'reg_via': torch.tensor(reg, dtype=torch.long),
            'genre': torch.tensor(genre, dtype=torch.long),
            'tab': torch.tensor(tab, dtype=torch.long),
            'screen': torch.tensor(screen, dtype=torch.long),
            'type': torch.tensor(typ, dtype=torch.long)
        }
        
        if self.is_train:
            return data, torch.tensor(self.targets[idx], dtype=torch.float)
        return data

# --- Model ---
class DeepFM(nn.Module):
    def __init__(self, n_users, n_items, cfg):
        super().__init__()
        self.emb_dim = cfg['EMBEDDING_DIM']
        
        self.user_emb = nn.Embedding(n_users, self.emb_dim)
        self.item_emb = nn.Embedding(n_items, self.emb_dim)
        
        self.tab_emb = nn.Embedding(50, self.emb_dim)
        self.screen_emb = nn.Embedding(50, self.emb_dim)
        self.type_emb = nn.Embedding(50, self.emb_dim)
        
        self.city_emb = nn.Embedding(30, self.emb_dim)
        self.gender_emb = nn.Embedding(5, self.emb_dim)
        self.reg_emb = nn.Embedding(10, self.emb_dim)
        self.genre_emb = nn.Embedding(2000, self.emb_dim) # Approx max genres
        
        # MLP
        # 9 fields
        mlp_in_dim = 9 * self.emb_dim 
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.final_linear = nn.Linear(64 + 1, 1) # +1 for FM

    def forward(self, data):
        u_emb = self.user_emb(data['user'])
        i_emb = self.item_emb(data['item'])
        tab_emb = self.tab_emb(data['tab'])
        scr_emb = self.screen_emb(data['screen'])
        typ_emb = self.type_emb(data['type'])
        cit_emb = self.city_emb(data['city'])
        gen_emb = self.gender_emb(data['gender'])
        reg_emb = self.reg_emb(data['reg_via'])
        gnr_emb = self.genre_emb(data['genre'])
        
        stacked_emb = torch.stack([
            u_emb, i_emb, tab_emb, scr_emb, typ_emb, cit_emb, gen_emb, reg_emb, gnr_emb
        ], dim=1)
        
        # FM
        sum_of_emb = torch.sum(stacked_emb, dim=1)
        sum_of_sq_emb = torch.sum(stacked_emb ** 2, dim=1)
        fm_out = 0.5 * (sum_of_emb ** 2 - sum_of_sq_emb)
        fm_out = torch.sum(fm_out, dim=1, keepdim=True)
        
        # Deep
        deep_emb = stacked_emb.view(stacked_emb.size(0), -1)
        mlp_out = self.mlp(deep_emb)
        
        # Combined
        final_input = torch.cat([mlp_out, fm_out], dim=1)
        logits = self.final_linear(final_input)
        
        return logits

def main():
    print("--- TRAIN DeepFM ---")
    
    # Load Data
    try:
        train = pd.read_csv(os.path.join(WORK_DIR, 'train.csv'), dtype={'target': np.int8})
        test = pd.read_csv(os.path.join(WORK_DIR, 'test.csv'))
        members = pd.read_csv(os.path.join(WORK_DIR, 'members_nn.csv')) # Use NN version
        songs = pd.read_csv(os.path.join(WORK_DIR, 'songs_nn.csv')) # Use NN version
    except FileNotFoundError:
        print("Error: Files not found. Run preprocess_data_export.py first.")
        return

    # Handle Categorical Mappings if they are not 0..N
    # Assuming preprocess_ids made them 0..N. 
    # But DeepFM needs to know N for Embeddings.
    
    n_users = len(members)
    n_items = len(songs)
    print(f"Users: {n_users}, Items: {n_items}")
    
    # Check max ID in train/test to ensure no out of bound
    max_uid = max(train['msno'].max(), test['msno'].max())
    max_iid = max(train['song_id'].max(), test['song_id'].max())
    
    if max_uid >= n_users:
        print(f"Warning: Max User ID {max_uid} >= User Count {n_users}. Resizing...")
        n_users = max_uid + 1
        
    if max_iid >= n_items:
        print(f"Warning: Max Song ID {max_iid} >= Song Count {n_items}. Resizing...")
        n_items = max_iid + 1

    cfg = {'EMBEDDING_DIM': 32, 'BATCH_SIZE': 2048, 'EPOCHS': 5}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Split
    X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)
    
    train_ds = NeuMFDataset(X_train, members, songs, is_train=True)
    val_ds = NeuMFDataset(X_val, members, songs, is_train=True)
    test_ds = NeuMFDataset(test, members, songs, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False)
    
    model = DeepFM(n_users, n_items, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training...")
    for epoch in range(cfg['EPOCHS']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            data, target = batch
            data = {k: v.to(device) for k, v in data.items()}
            target = target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
    print("Predicting...")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch
            data = {k: v.to(device) for k, v in data.items()}
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            preds.extend(probs)
            
    sub = pd.DataFrame({'id': test['id'], 'target': preds})
    sub.to_csv('submission_deepfm.csv', index=False)
    print("Saved submission_deepfm.csv")

if __name__ == "__main__":
    main()
