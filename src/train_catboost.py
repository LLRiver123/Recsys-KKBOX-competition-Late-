import catboost as cb
import pandas as pd
import numpy as np
import gc
import os
from sklearn.model_selection import train_test_split
from config import WORK_DIR

def main():
    print("--- TRAIN CATBOOST ---")

    # 1. Load Data
    try:
        train = pd.read_csv(os.path.join(WORK_DIR, 'train.csv'), dtype={'target': np.int8})
        test = pd.read_csv(os.path.join(WORK_DIR, 'test.csv'))
        members = pd.read_csv(os.path.join(WORK_DIR, 'members_gbdt.csv'))
        songs = pd.read_csv(os.path.join(WORK_DIR, 'songs_gbdt.csv'))
    except FileNotFoundError:
        print("Error: Prepared files not found.")
        return

    print("Merging features...")
    train = train.merge(members, on='msno', how='left')
    train = train.merge(songs, on='song_id', how='left')
    
    test = test.merge(members, on='msno', how='left')
    test = test.merge(songs, on='song_id', how='left')

    del members, songs
    gc.collect()

    target = train['target']
    submission_ids = test['id']

    # Define categorical columns
    # These should exist in the merged dataframe
    cat_cols_names = [
        'msno', 'song_id', 'source_system_tab', 'source_screen_name', 
        'source_type', 'city', 'gender', 'registered_via', 'language', 
        'artist_name', 'first_genre_id', 'second_genre_id', 'third_genre_id', 
        'is_featured'
    ]
    
    # Filter valid columns
    valid_cat_cols = [c for c in cat_cols_names if c in train.columns]
    
    print("Fixing categorical columns...")
    # Fix types for CatBoost
    for col in valid_cat_cols:
        train[col] = train[col].fillna(-1).astype(int).astype(str)
        test[col] = test[col].fillna(-1).astype(int).astype(str)

    # Drop unused
    drop_cols = ['target', 'id', 'registration_init_time', 'expiration_date']
    features = [c for c in train.columns if c not in drop_cols]
    
    # Get Cat indices
    cat_features_indices = [features.index(c) for c in valid_cat_cols if c in features]
    print(f"Categorical Indices: {cat_features_indices}")

    X_train_full = train[features]
    X_test = test[features]

    del train, test
    gc.collect()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, target, test_size=0.2, random_state=42
    )

    train_pool = cb.Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = cb.Pool(X_val, y_val, cat_features=cat_features_indices)
    test_pool = cb.Pool(X_test, cat_features=cat_features_indices)

    # Params
    params = {
        'learning_rate': 0.1, # Adjusted for CPU/Speed
        'depth': 8,
        'l2_leaf_reg': 7.84,
        'random_strength': 0.612,
        'bagging_temperature': 0.359,
        'iterations': 1000, 
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        # 'task_type': 'GPU', # Uncomment if GPU
        # 'devices': '0',
        'verbose': 100,
        'early_stopping_rounds': 50
    }

    print("Training CatBoost...")
    model = cb.CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    print("Predicting...")
    preds = model.predict_proba(test_pool)[:, 1]

    sub = pd.DataFrame({'id': submission_ids, 'target': preds})
    sub['id'] = sub['id'].astype(int)
    
    filename = 'submission_catboost.csv'
    sub.to_csv(filename, index=False)
    print(f"Saved {filename}")

if __name__ == "__main__":
    main()
