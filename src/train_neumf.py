import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gc
import os
import sys

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not found. SBERT features will be disabled/random.")

from config import WORK_DIR, INPUT_DIR

# --- CONFIGURATION ---
CONFIG = {
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'DATA_DIR': INPUT_DIR, # Using INPUT_DIR from config
    'BATCH_SIZE': 2048,
    'EPOCHS': 15,
    'LR': 0.001,
    'EMBEDDING_DIM': 64,
    'SBERT_DIM': 384,
    'TEXT_PROJ_DIM': 32,
    'PATIENCE': 3
}

def load_data_optimized():
    dtypes = {
        'msno': 'object', 'song_id': 'object', 
        'source_system_tab': 'object', 'source_screen_name': 'object', 'source_type': 'object',
        'target': 'uint8',
        'song_length': 'float32',
        'artist_name': 'object', 'composer': 'object', 'lyricist': 'object',
        'city': 'object',
        'gender': 'object',
        'registered_via': 'object',
        'bd': 'int16'
    }

    print("Loading csv files...")
    # Attempt to load from INPUT_DIR
    try:
        train = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'train.csv'), dtype=dtypes)
        test = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'test.csv'), dtype=dtypes)
        songs = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'songs.csv'), dtype=dtypes)
        members = pd.read_csv(os.path.join(CONFIG['DATA_DIR'], 'members.csv'), dtype=dtypes)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure train.csv, test.csv, songs.csv, members.csv are in the project root.")
        sys.exit(1)
    
    print("Merging data...")
    train = train.merge(songs, on='song_id', how='left')
    train = train.merge(members, on='msno', how='left')
    test = test.merge(songs, on='song_id', how='left')
    test = test.merge(members, on='msno', how='left')
    
    del songs, members
    gc.collect()
    return train, test

def preprocess_data(train, test):
    print("Handling NaNs...")
    str_cols = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 
                'artist_name', 'composer', 'lyricist', 'city', 'gender', 'registered_via']
    
    for col in str_cols:
        if col in train.columns:
            train[col] = train[col].fillna('unknown').astype(str)
            test[col] = test[col].fillna('unknown').astype(str)

    # Process Age (bd)
    valid_idx = (train['bd'] >= 10) & (train['bd'] <= 80)
    mean_age = int(train.loc[valid_idx, 'bd'].mean())
    
    train.loc[~valid_idx, 'bd'] = mean_age
    test.loc[(test['bd'] < 10) | (test['bd'] > 80), 'bd'] = mean_age

    print("Creating Text Features...")
    train['text_feat'] = train['artist_name'] + " " + train['composer'] + " " + train['lyricist']
    test['text_feat'] = test['artist_name'] + " " + test['composer'] + " " + test['lyricist']

    print("Encoding Categoricals...")
    enc_cols = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 
                'city', 'gender', 'registered_via']
    
    encoders = {}
    for col in tqdm(enc_cols):
        if col in train.columns:
            le = LabelEncoder()
            full_vals = pd.concat([train[col], test[col]]).unique()
            le.fit(full_vals)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            encoders[col] = le
        
    return train, test, encoders

def precompute_sbert_embeddings(df_list, n_items):
    if not SBERT_AVAILABLE:
        print("SBERT not available. Generating random embeddings.")
        return np.random.rand(n_items, CONFIG['SBERT_DIM'])

    print("Loading SBERT Model...")
    try:
        sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=str(CONFIG['DEVICE']))
    except Exception as e:
        print(f"Failed to load SBERT model: {e}")
        return np.random.rand(n_items, CONFIG['SBERT_DIM'])
   
    print("Extracting unique songs...")
    all_df = pd.concat(df_list)[['song_id', 'text_feat']].drop_duplicates('song_id')
    
    # song_id is label encoded 0..N
    # We need an array where index matches song_id
    max_id = all_df['song_id'].max()
    # n_items should be max_id + 1
    
    song_texts = ["unknown"] * n_items
    for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
        sid = int(row['song_id'])
        if sid < n_items:
            song_texts[sid] = str(row['text_feat'])
        
    print("Encoding songs with SBERT...")
    embeddings = sbert.encode(
        song_texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    del sbert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return embeddings

class NeuMFDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.users = df['msno'].values
        self.items = df['song_id'].values
        
        # Context
        self.tabs = df['source_system_tab'].values
        self.screens = df['source_screen_name'].values
        self.types = df['source_type'].values
        
        # User Meta
        self.cities = df['city'].values
        self.genders = df['gender'].values
        self.reg_via = df['registered_via'].values
        
        self.is_train = is_train
        if is_train:
            self.targets = df['target'].values
            
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, i):
        data = {
            'user': torch.tensor(self.users[i], dtype=torch.long),
            'item': torch.tensor(self.items[i], dtype=torch.long),
            
            # Context
            'tab': torch.tensor(self.tabs[i], dtype=torch.long),
            'screen': torch.tensor(self.screens[i], dtype=torch.long),
            'type': torch.tensor(self.types[i], dtype=torch.long),
            
            # User Meta
            'city': torch.tensor(self.cities[i], dtype=torch.long),
            'gender': torch.tensor(self.genders[i], dtype=torch.long),
            'reg_via': torch.tensor(self.reg_via[i], dtype=torch.long),
        }
        
        if self.is_train:
            return data, torch.tensor(self.targets[i], dtype=torch.float)
        return data

class HybridNeuMF_Full(nn.Module):
    def __init__(self, n_users, n_items, pretrained_sbert, cfg):
        super().__init__()
        dim = cfg['EMBEDDING_DIM']
        
        # 1. GMF (Dot Product)
        self.gmf_user = nn.Embedding(n_users, dim)
        self.gmf_item = nn.Embedding(n_items, dim)
        
        # 2. MLP User Part (User Offset)
        self.mlp_user = nn.Embedding(n_users, dim)
        
        # User Metadata Embeddings
        self.city_emb = nn.Embedding(30, 8)
        self.gender_emb = nn.Embedding(5, 4)
        self.reg_emb = nn.Embedding(10, 4)
        self.meta_proj = nn.Linear(8+4+4, dim) 
        
        # 3. MLP Item Part (SBERT)
        self.mlp_item = nn.Embedding(n_items, dim)
        self.sbert_emb = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_sbert), freeze=True)
        self.sbert_proj = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(),
            nn.Linear(128, cfg['TEXT_PROJ_DIM']) 
        )
        
        # 4. Context Part
        self.tab_emb = nn.Embedding(50, 8)
        self.screen_emb = nn.Embedding(50, 8)
        self.type_emb = nn.Embedding(50, 8)
        
        # 5. Fusion MLP
        # Input: User(64) + Item(64) + SBERT(32) + Context(24) = 184
        mlp_in_dim = dim * 2 + cfg['TEXT_PROJ_DIM'] + 24
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        
        # Final: GMF(64) + MLP(32)
        self.final = nn.Linear(dim + 32, 1)

    def forward(self, data):
        # GMF Branch
        u_gmf = self.gmf_user(data['user'])
        i_gmf = self.gmf_item(data['item'])
        gmf_out = u_gmf * i_gmf
        
        # MLP Branch - User (Offset Trick)
        u_mlp = self.mlp_user(data['user'])
        # Metadata
        # Clamp inputs to avoid out of range if new categories appear
        # Assuming fixed embedding sizes (30, 5, 10), we clip indices.
        c_city = torch.clamp(data['city'], 0, 29)
        c_gender = torch.clamp(data['gender'], 0, 4)
        c_reg = torch.clamp(data['reg_via'], 0, 9)
        
        meta = torch.cat([
            self.city_emb(c_city), 
            self.gender_emb(c_gender), 
            self.reg_emb(c_reg)
        ], dim=1)
        u_meta = self.meta_proj(meta)
        u_vec = u_mlp + u_meta 
        
        # MLP Branch - Item (SBERT)
        i_mlp = self.mlp_item(data['item'])
        txt = self.sbert_proj(self.sbert_emb(data['item']))
        
        # Context
        # Clamp context
        c_tab = torch.clamp(data['tab'], 0, 49)
        c_screen = torch.clamp(data['screen'], 0, 49)
        c_type = torch.clamp(data['type'], 0, 49)

        ctx = torch.cat([
            self.tab_emb(c_tab),
            self.screen_emb(c_screen),
            self.type_emb(c_type)
        ], dim=1)
        
        # Concatenate All for MLP
        mlp_in = torch.cat([u_vec, i_mlp, txt, ctx], dim=1)
        mlp_out = self.mlp(mlp_in)
        
        # Final Fusion
        out = self.final(torch.cat([gmf_out, mlp_out], dim=1))
        return out

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_score = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_auc, model):
        if val_auc > self.best_score:
            self.best_score = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def main():
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # 1. Load Data
    train_df, test_df = load_data_optimized()
    
    # 2. Preprocess
    train_df, test_df, encoders = preprocess_data(train_df, test_df)
    
    N_USERS = len(encoders['msno'].classes_)
    N_ITEMS = len(encoders['song_id'].classes_)
    print(f"Num Users: {N_USERS}, Num Items: {N_ITEMS}")
    
    # 3. SBERT
    sbert_embeddings = precompute_sbert_embeddings([train_df, test_df], N_ITEMS)
    
    # 4. Split Train/Val
    # Split 20% from train for validation
    train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_split)}, Val size: {len(val_split)}")
    
    # 5. Datasets & Loaders
    train_ds = NeuMFDataset(train_split, is_train=True)
    val_ds = NeuMFDataset(val_split, is_train=True)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0) # workers 0 for safety
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    # 6. Model
    model = HybridNeuMF_Full(N_USERS, N_ITEMS, sbert_embeddings, CONFIG).to(CONFIG['DEVICE'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    stopper = EarlyStopping(patience=CONFIG['PATIENCE'])
    
    print("--- START TRAINING ---")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            data = {k: v.to(CONFIG['DEVICE']) for k, v in data.items()}
            target = target.to(CONFIG['DEVICE']).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        targets, preds = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data = {k: v.to(CONFIG['DEVICE']) for k, v in data.items()}
                output = model(data)
                
                preds.extend(torch.sigmoid(output).cpu().numpy())
                targets.extend(target.cpu().numpy())
                
        val_auc = roc_auc_score(targets, preds)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        scheduler.step(val_auc)
        stopper(val_auc, model)
        if stopper.early_stop:
            print(f"Early stopping triggered! Best AUC: {stopper.best_score:.4f}")
            break
            
    # 7. Prediction
    print("Predicting Test Set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_ds = NeuMFDataset(test_df, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE']*2, shuffle=False, num_workers=0)
    
    all_preds = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting"):
            data = {k: v.to(CONFIG['DEVICE']) for k, v in data.items()}
            output = model(data)
            all_preds.extend(torch.sigmoid(output).cpu().numpy().flatten())
            
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': all_preds
    })
    
    submission.to_csv('submission_neumf.csv', index=False)
    print("Submission saved to submission_neumf.csv")

if __name__ == "__main__":
    main()
