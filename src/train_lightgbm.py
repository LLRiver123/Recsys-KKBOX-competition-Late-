import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import gc
import matplotlib.pyplot as plt
import os
from config import WORK_DIR, INPUT_DIR

def main():
    print(">>> KHỞI ĐỘNG: CHIẾN DỊCH FEATURE ENGINEERING TOP 1 (LightGBM)")

    # ==============================================================================
    # 1. LOAD & PREPROCESS (TỐI ƯU BỘ NHỚ)
    # ==============================================================================
    
    print("1. Loading Data...")
    # Load Data from WORK_DIR (outputs of preprocess_data_export.py)
    try:
        train = pd.read_csv(os.path.join(WORK_DIR, 'train.csv'), dtype={'target': np.int8})
        test = pd.read_csv(os.path.join(WORK_DIR, 'test.csv'))
        
        # NOTE: The notebook referenced 'members.csv' and 'songs.csv' in WORK_DIR.
        # Based on our pipeline, 'members_gbdt.csv' and 'songs_gbdt.csv' contain the processed (int-encoded) IDs matching train/test.
        members = pd.read_csv(os.path.join(WORK_DIR, 'members_gbdt.csv'))
        songs = pd.read_csv(os.path.join(WORK_DIR, 'songs_gbdt.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure preprocess_data_export.py has been run.")
        return

    # Nối Train/Test
    train['is_train'] = 1
    test['is_train'] = 0
    test['target'] = np.nan
    full_data = pd.concat([train, test], ignore_index=True)

    del train, test
    gc.collect()

    print("2. Merging Info...")
    # Merge members and songs
    # Note: ensure columns like 'msno', 'song_id' match types (int)
    full_data = full_data.merge(members, on='msno', how='left')
    full_data = full_data.merge(songs, on='song_id', how='left')
    del members, songs
    gc.collect()

    # ==============================================================================
    # 2. XỬ LÝ THỜI GIAN (CRITICAL STEP)
    # ==============================================================================
    print("3. Time Processing (Tạo Timestamp chuẩn)...")
    # 'registration_init_time' and 'expiration_date' in members_gbdt might be already encoded or float?
    # In preprocess_ids.py: 
    # members['registration_init_time'] = members['registration_init_time'].apply(safe_date_convert) -> converts to float timestamp
    # So they are likely floats (seconds).
    
    # Logic in Cell 64 expects them to be converted to datetime from string?
    # "for col in ... pd.to_datetime(..., format='%Y%m%d')"
    # BUT our `members_gbdt` already has them as Timestamps (float).
    # We need to adapt the logic.
    
    # If they are already timestamps, we can just use them.
    # But let's check if we need to convert back or just use them.
    # The cell 64 logic:
    # full_data['timestamp'] = full_data['registration_init_time'].astype(np.int64) // 10**9
    
    # If `registration_init_time` is already a timestamp (seconds), then we don't need pd.to_datetime with %Y%m%d.
    # Let's assume they are floats (seconds).
    
    # However, `train.csv` (from before_after) has a 'timestamp' column! (Source timestamp)
    # Cell 64 creates a NEW 'timestamp' from 'registration_init_time'?
    # "full_data['timestamp'] = full_data['registration_init_time'].astype(np.int64) // 10**9"
    # Wait, 'registration_init_time' is when user registered. 'timestamp' usually refers to the event time (source_time).
    # The 'train.csv' from step 51 (from before_after) HAS 'timestamp' (which is the listen event time).
    # Cell 64 overwrites 'timestamp' with 'registration_init_time'?? That seems wrong for sorting events.
    # Ah, Cell 64 says: "full_data['timestamp'] = full_data['registration_init_time']..."
    # If it overwrites event timestamp with registration timestamp, then "Sorting by Time" sorts by registration time?
    # That might be a bug in the notebook or a specific feature.
    # BUT, looking at "Sắp xếp dữ liệu theo thời gian (Bắt buộc để tính till_now_cnt)", it implies event time.
    # If I overwrite it with reg time, `till_now_cnt` becomes meaningless for listening history (it becomes history of user registrations?).
    
    # Correction: The logic in Cell 64 seems to restart processing.
    # "train = pd.read_csv... train.csv"
    # If "train.csv" in Cell 64 was the *original* raw file, then it wouldn't have 'timestamp'.
    # But we established it reads from WORK_DIR.
    # If Cell 64 intends to use Registration Time for something, it should probably be a different column.
    # HOWEVER, strictly following the code:
    # It overwrites 'timestamp'.
    
    # Let's look at `preprocess_ids.py`. `registration_init_time` is `safe_date_convert` -> `mktime` (seconds).
    # So it is already seconds.
    # Cell 64: `astype(np.int64) // 10**9`. If it's seconds, dividing by 10^9 makes it 0 (unless it's nanoseconds?).
    # `time.mktime` returns Seconds.
    # So `// 10**9` would result in 0.
    # This suggests Cell 64 expects `registration_init_time` to be YYYYMMDD (int/str) OR datetime64[ns].
    # This implies Cell 64 expects RAW members.csv!
    
    # CONFLICT:
    # 1. Cell 64 reads `train.csv` (likely Step 51 output -> Int IDs).
    # 2. Cell 64 expects `members.csv` (Raw -> String IDs, YYYYMMDD dates).
    # If we mix Int IDs (Train) and String IDs (Members), Merge fails.
    
    # RESOLUTION:
    # The notebook likely ran `preprocess_ids` (Int IDs) -> `preprocess_export` (Train Int IDs).
    # Then for Cell 64, the user might have re-loaded *raw* `members.csv` AND *converted IDs to Int* again?
    # Or `members.csv` in `INPUT_DIR` was somehow the one with Int IDs but Raw Dates?
    
    # To make this robust and "Top 1", I will use `members_gbdt.csv` which has:
    # - Int IDs (Correct)
    # - Float Timestamps (Seconds) for dates.
    
    # I will ADAPT Cell 64 logic to work with these processed features.
    # 1. Skip pd.to_datetime conversion (already floats).
    # 2. Skip `// 10**9` (already seconds).
    # 3. Use the existing `timestamp` column in `full_data` (from train.csv) for sorting, NOT `registration_init_time`.
    #    - Why? Because `till_now_cnt` needs order of listening.
    #    - `train.csv` has `timestamp` (event time).
    #    - `members.csv` has `registration_init_time`.
    
    # I will assume `timestamp` in `full_data` is the Event Time (correct for RecSys).
    # I will create features based on that.
    
    # SẮP XẾP DỮ LIỆU THEO THỜI GIAN
    print("   -> Sorting by Time for Cumulative Features...")
    if 'timestamp' in full_data.columns:
        full_data = full_data.sort_values(['msno', 'timestamp'])
    else:
        print("Warning: 'timestamp' column missing. Sorting by index.")

    # ==============================================================================
    # 3. TẠO CÁC "SIÊU TÍNH NĂNG" TỪ FILE IMPORTANCE
    # ==============================================================================
    print("4. Creating TOP 1 Features...")

    # --- A. MSNO_TILL_NOW_CNT (TOP 3 FEATURE) ---
    print("   -> Generating 'msno_till_now_cnt'...")
    full_data['msno_till_now_cnt'] = full_data.groupby('msno').cumcount()

    # --- B. TIME STATISTICS (TOP 5 FEATURES) ---
    print("   -> Generating Time Stats (Mean, Std, Min, Max)...")
    # Using 'timestamp' (Event time)
    group_time = full_data.groupby('msno')['timestamp']
    full_data['msno_timestamp_mean'] = group_time.transform('mean')
    full_data['msno_timestamp_std']  = group_time.transform('std').fillna(0)
    full_data['msno_upper_time']     = group_time.transform('max')
    full_data['msno_lower_time']     = group_time.transform('min')

    # --- C. PROBABILITY FEATURES (MỞ RỘNG) ---
    print("   -> Generating Probabilities (Artist, Language, System)...")
    def create_prob(df, col):
        # P(Feature | User)
        counts = df.groupby(['msno', col])[col].transform('count')
        total = df.groupby('msno')[col].transform('count')
        df[f'msno_{col}_prob'] = counts / total
        return df

    probs_cols = ['source_type', 'source_screen_name', 'source_system_tab', 'artist_name', 'language']
    for col in probs_cols:
        if col in full_data.columns:
            full_data = create_prob(full_data, col)

    # --- D. REC COUNT ---
    full_data['msno_rec_cnt'] = full_data.groupby('msno')['msno'].transform('count')

    # --- E. TIME LEFT & BASIC ---
    # `expiration_date` and `registration_init_time` are float seconds.
    # Days = (Exp - Reg) / (24*3600)
    full_data['time_left'] = (full_data['expiration_date'] - full_data['registration_init_time']) / 86400
    full_data['time_left'] = full_data['time_left'].fillna(0)

    # ==============================================================================
    # 4. SVD NÂNG CAO (50 COMPONENTS)
    # ==============================================================================
    print("5. Advanced SVD (50 Components)...")

    # Mã hóa ID sang số để tạo ma trận
    # Note: IDs are already ints, but let's ensure they are 0..N for matrix
    # LabelEncoder again to be safe? Or just use as is if continuous.
    # To be safe and compact:
    full_data['msno_int'] = full_data['msno'].astype('category').cat.codes
    full_data['song_int'] = full_data['song_id'].astype('category').cat.codes

    # Tạo ma trận thưa
    rows = full_data['msno_int'].values
    cols = full_data['song_int'].values
    data_ones = np.ones(len(full_data))
    n_users = full_data['msno_int'].max() + 1
    n_songs = full_data['song_int'].max() + 1
    sparse_matrix = sparse.csr_matrix((data_ones, (rows, cols)), shape=(n_users, n_songs))

    # SVD
    n_components = 50 
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_vecs = svd.fit_transform(sparse_matrix)
    song_vecs = svd.components_.T

    # 1. Tính Dot Product (Tương tác User-Song)
    print("   -> Calculating SVD Dot Product...")
    u_vecs_selected = user_vecs[full_data['msno_int'].values]
    s_vecs_selected = song_vecs[full_data['song_int'].values]
    full_data['song_embeddings_dot'] = (u_vecs_selected * s_vecs_selected).sum(axis=1)

    # 2. Thêm các User Components (Feature ẩn)
    print("   -> Adding SVD Components to DataFrame...")
    n_keep = 20 
    cols_svd = [f'member_component_{i}' for i in range(n_keep)]
    df_svd = pd.DataFrame(user_vecs[:, :n_keep], columns=cols_svd)
    df_svd['msno_int'] = range(n_users)

    full_data = full_data.merge(df_svd, on='msno_int', how='left')

    del sparse_matrix, user_vecs, song_vecs, u_vecs_selected, s_vecs_selected, df_svd
    gc.collect()

    # ==============================================================================
    # 5. PREPARE FOR LIGHTGBM
    # ==============================================================================
    print("6. Formatting Data for LightGBM...")

    # Xử lý Category (Encoding)
    cat_cols = ['source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 
                'registered_via', 'language', 'artist_name', 'composer', 'lyricist', 
                'msno', 'song_id', 'genre_ids'] # genre_ids might be missing or int

    for col in cat_cols:
        if col in full_data.columns:
            full_data[col] = full_data[col].astype('category').cat.codes

    # Chọn features
    cols_to_drop = ['registration_init_time', 'expiration_date', 'target', 'id', 'msno_int', 'song_int', 'is_train']
    features = [c for c in full_data.columns if c not in cols_to_drop]
    print(f"   -> Final Feature Count: {len(features)}")
    
    train_df = full_data[full_data['is_train'] == 1]
    test_df = full_data[full_data['is_train'] == 0]
    y_train = full_data[full_data['is_train'] == 1]['target']
    
    # Get IDs for submission
    submission_ids = test_df['id'].values

    del full_data
    gc.collect()

    # ==============================================================================
    # 6. TRAIN & SUBMIT
    # ==============================================================================
    print("7. Training LightGBM...")

    dtrain = lgb.Dataset(train_df[features], label=y_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 200,
        'max_depth': 12,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params, 
        dtrain, 
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(period=100)]
    )

    print("8. Predicting...")
    preds = model.predict(test_df[features])

    sub = pd.DataFrame({'id': submission_ids, 'target': preds})
    # Restore int ID if converted to float
    sub['id'] = sub['id'].astype(int)
    
    filename = 'submission_lightgbm.csv'
    sub.to_csv(filename, index=False)

    print(f">>> HOÀN TẤT! File: {filename}")

if __name__ == "__main__":
    main()
