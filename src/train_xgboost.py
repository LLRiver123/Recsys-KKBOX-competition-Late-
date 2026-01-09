import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import os
from sklearn.model_selection import train_test_split
from config import WORK_DIR

def main():
    print("--- TRAIN XGBOOST ---")

    # 1. Load Data
    print("Loading prepared data...")
    try:
        train = pd.read_csv(os.path.join(WORK_DIR, 'train.csv'), dtype={'target': np.int8})
        test = pd.read_csv(os.path.join(WORK_DIR, 'test.csv'))
        # Load members/songs if needed for merging additional features
        members = pd.read_csv(os.path.join(WORK_DIR, 'members_gbdt.csv'))
        songs = pd.read_csv(os.path.join(WORK_DIR, 'songs_gbdt.csv'))
    except FileNotFoundError:
        print("Error: Prepared files not found. Run preprocess_data_export.py first.")
        return

    # Merge features
    print("Merging features...")
    train = train.merge(members, on='msno', how='left')
    train = train.merge(songs, on='song_id', how='left')
    
    test = test.merge(members, on='msno', how='left')
    test = test.merge(songs, on='song_id', how='left')

    del members, songs
    gc.collect()

    # Drop non-feature columns
    target = train['target']
    drop_cols = ['target', 'id', 'msno', 'song_id', 'registration_init_time', 'expiration_date']
    # Note: timestamps might be useful, keep 'timestamp' (event time) if exists
    
    # Identify feature columns
    # Exclude columns that are definitely not features
    features = [c for c in train.columns if c not in drop_cols]
    
    print(f"Features: {len(features)}")
    
    X_train = train[features]
    X_test = test[features]
    submission_ids = test['id']

    del train, test
    gc.collect()

    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, target, test_size=0.2, random_state=42
    )

    print("Creating DMatrix...")
    dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
    dval = xgb.DMatrix(X_val_split, label=y_val_split)
    dtest = xgb.DMatrix(X_test)

    # Params (from Cell 30)
    params = {
        'objective': 'binary:logistic',
        # 'tree_method': 'gpu_hist',  # Uncomment if GPU available
        'tree_method': 'hist',        # CPU version
        'random_state': 42,
        'eval_metric': ['logloss', 'auc'], 
        'learning_rate': 0.0973,
        'max_depth': 11,
        'min_child_weight': 8,
        'subsample': 0.879,
        'colsample_bytree': 0.763,
        'gamma': 0.174,
        'lambda': 0.0093,
        'alpha': 0.0022
    }

    print("Training XGBoost...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000, # Reduced from 3000 for standard run
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    print("Predicting...")
    preds = model.predict(dtest)

    sub = pd.DataFrame({'id': submission_ids, 'target': preds})
    sub['id'] = sub['id'].astype(int)
    
    filename = 'submission_xgboost.csv'
    sub.to_csv(filename, index=False)
    print(f"Saved {filename}")

if __name__ == "__main__":
    main()
