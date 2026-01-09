import numpy as np
import pandas as pd
import os
import gc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder
from config import TEMP_DIR, INPUT_DIR

def main():
    print("\n--- Đang chạy: preprocess_counts.py ---")

    # ==============================================================================
    # PHẦN 1: COUNT LOG PROCESS (Tính toán thống kê từ file ID đã lưu)
    # ==============================================================================
    
    # 1. Load Data
    try:
        train = pd.read_csv(os.path.join(TEMP_DIR, 'train_id.csv'))
        test = pd.read_csv(os.path.join(TEMP_DIR, 'test_id.csv'))
        member = pd.read_csv(os.path.join(TEMP_DIR, 'members_id.csv'))
        song = pd.read_csv(os.path.join(TEMP_DIR, 'songs_id.csv'))
    except FileNotFoundError:
        print("Error: Required '_id.csv' files not found in temporary directory.")
        print("Please run preprocess_ids.py first.")
        return

    # Song DataFrame chuẩn hóa
    song_base = pd.DataFrame({'song_id': range(max(train.song_id.max(), test.song_id.max())+1)})
    song = song_base.merge(song, on='song_id', how='left')

    data = pd.concat([train[['msno', 'song_id']], test[['msno', 'song_id']]])

    # 2. Member Count
    print("Generating Member Counts...")
    mem_rec_cnt = data['msno'].value_counts().to_dict()
    member['msno_rec_cnt'] = member['msno'].map(mem_rec_cnt).fillna(0)
    member['bd'] = member['bd'].apply(lambda x: np.nan if x <= 0 or x >= 75 else x)

    # 3. Song Counts (Artist, Composer, Genre...)
    print("Generating Song Counts...")
    for col in ['artist_name', 'composer', 'lyricist', 'first_genre_id']:
        cnt_map = song[col].value_counts().to_dict()
        song[f'{col.replace("_name","")}_song_cnt'] = song[col].map(cnt_map).fillna(0)

    # 4. Rec Counts (Số lần được nghe)
    # Merge thông tin bài hát vào bảng tương tác (data) để đếm
    data_merged = data.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', 'first_genre_id']], on='song_id', how='left')

    for col in ['song_id', 'artist_name', 'composer', 'lyricist', 'first_genre_id']:
        rec_cnt_map = data_merged[col].value_counts().to_dict()
        
        # Đặt tên cột output
        prefix = col.replace('_name', '')
        if col == 'first_genre_id': prefix = 'genre'
        
        song[f'{prefix}_rec_cnt'] = song[col].map(rec_cnt_map).fillna(0)

    # 5. MSNO Context Features (Mean Encoding cho Context)
    print("Generating Context Features...")
    dummy_feat = ['source_system_tab', 'source_screen_name', 'source_type']
    concat_df = pd.concat([train.drop('target', axis=1), test.drop('id', axis=1)])

    for feat in dummy_feat:
        # Get dummies
        dummies = pd.get_dummies(concat_df[feat].astype(str), prefix=f'msno_{feat}')
        dummies['msno'] = concat_df['msno'].values
        
        # Group mean
        feat_profile = dummies.groupby('msno').mean().reset_index()
        member = member.merge(feat_profile, on='msno', how='left')

    # 6. Log Transform & Save
    print("Log Transforming & Saving...")
    features = ['msno_rec_cnt']
    for feat in features:
        member[feat] = np.log1p(member[feat])
    member.to_csv(os.path.join(TEMP_DIR, 'members_id_cnt.csv'), index=False)

    song_log_feats = ['song_length', 'song_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', 
                      'lyricist_song_cnt', 'genre_song_cnt', 'artist_rec_cnt', 
                      'composer_rec_cnt', 'lyricist_rec_cnt', 'genre_rec_cnt']

    for feat in song_log_feats:
        if feat in song.columns:
            song[feat] = song[feat].fillna(0).clip(lower=0)
            song[feat] = np.log1p(song[feat])

    song.to_csv(os.path.join(TEMP_DIR, 'songs_id_cnt.csv'), index=False)
    train.to_csv(os.path.join(TEMP_DIR, 'train_id_cnt.csv'), index=False)
    test.to_csv(os.path.join(TEMP_DIR, 'test_id_cnt.csv'), index=False)

    print(f"Hoàn thành Count Process! File đã lưu tại: {TEMP_DIR}")

    # ==================================================
    # PHẦN 3: ISRC PROCESSING (FIXED)
    # ==================================================

    print("==================================================")
    print("PHẦN 3: ISRC PROCESSING (FIXED)")
    print("==================================================")

    # 1. Load data (Using data from memory if possible, else reload)
    # We already have song in memory, but it's modified. Let's work with 'song' df.
    # train and test also in memory.
    
    # 2. Xử lý ISRC
    print("Processing ISRC features...")

    # Tạo cột isrc mặc định (Dummy) để tránh crash nếu file thiếu
    if 'isrc' not in song.columns:
        print("   -> Cột 'isrc' thiếu. Đang tạo Dummy ISRC...")
        song['isrc'] = np.nan

    # Fill NaN
    isrc = song['isrc'].fillna('000000000000').astype(str)

    # Cắt chuỗi
    song['cc'] = isrc.str.slice(0, 2)
    song['xxx'] = isrc.str.slice(2, 5)

    # Xử lý năm (YY)
    def parse_year(x):
        try:
            val = int(x)
            return 2000 + val if val < 18 else 1900 + val
        except:
            return 2017 # Default year

    song['yy'] = isrc.str.slice(5, 7).apply(parse_year)

    # Encode
    print("Encoding ISRC parts...")
    song['cc'] = LabelEncoder().fit_transform(song['cc'])
    song['xxx'] = LabelEncoder().fit_transform(song['xxx'])
    song['isrc_missing'] = (song['cc'] == 0).astype(int)

    # 3. Tạo Count Features cho ISRC
    print("Generating ISRC Counts...")
    for col in ['cc', 'xxx', 'yy']:
        cnt_map = song[col].value_counts().to_dict()
        song[f'{col}_song_cnt'] = song[col].map(cnt_map).fillna(0)

    # Merge với data tương tác để đếm Rec Count
    # Lưu ý: Chỉ lấy các cột cần thiết để tiết kiệm RAM
    # 'data' df from previous step still has msno, song_id
    data = data.merge(song[['song_id', 'cc', 'xxx', 'yy']], on='song_id', how='left')

    for col in ['cc', 'xxx', 'yy']:
        rec_map = data[col].value_counts().to_dict()
        song[f'{col}_rec_cnt'] = song[col].map(rec_map).fillna(0)

    # Log transform & Clean up
    new_feats = ['cc_song_cnt', 'xxx_song_cnt', 'yy_song_cnt', 'cc_rec_cnt', 'xxx_rec_cnt', 'yy_rec_cnt']
    for feat in new_feats:
        song[feat] = np.log1p(np.maximum(song[feat], 0))

    if 'isrc' in song.columns:
        song.drop('isrc', axis=1, inplace=True)

    # Lưu kết quả
    song.to_csv(os.path.join(TEMP_DIR, 'songs_id_cnt_isrc.csv'), index=False)
    print("Saved: songs_id_cnt_isrc.csv")


    print("\n==================================================")
    print("PHẦN 4: SVD PROCESSING (FIXED TYPE ERROR)")
    print("==================================================")

    # 1. Chuẩn bị dữ liệu
    print("Preparing SVD Matrix...")
    # members is already in memory as 'member'
    # song is already in memory as 'song'
    
    # Gộp toàn bộ tương tác
    df_all = pd.concat([train, test], axis=0).reset_index(drop=True)
    # Merge Artist ID vào df_all
    df_all = df_all.merge(song[['song_id', 'artist_name']], on='song_id', how='left')

    # 2. SVD User-Song
    print("Running SVD on User-Song...")
    n_components = 48 

    # [FIX]: Ép kiểu int cho kích thước ma trận
    n_users = int(member['msno'].max() + 1)
    n_items = int(song['song_id'].max() + 1)

    row = df_all['msno'].fillna(0).astype(int).values
    col = df_all['song_id'].fillna(0).astype(int).values
    data_ones = np.ones(len(df_all))

    # Tạo ma trận thưa
    R_song = coo_matrix((data_ones, (row, col)), shape=(n_users, n_items))

    # SVD Calculation
    u_song, s_song, vt_song = svds(R_song.astype(float), k=n_components)

    # Save Latent Factors (Member)
    u_cols = [f'member_component_{i}' for i in range(n_components)]
    members_svd = pd.DataFrame(u_song, columns=u_cols)
    members_svd['msno'] = range(n_users)
    member = member.merge(members_svd, on='msno', how='left')

    # Save Latent Factors (Song)
    v_cols = [f'song_component_{i}' for i in range(n_components)]
    songs_svd = pd.DataFrame(vt_song.T, columns=v_cols)
    songs_svd['song_id'] = range(n_items)
    song = song.merge(songs_svd, on='song_id', how='left')

    # 3. SVD User-Artist
    print("Running SVD on User-Artist...")
    n_components_art = 16

    # [FIX]: Ép kiểu int cho Artist ID và kích thước ma trận
    df_all['artist_name'] = df_all['artist_name'].fillna(0).astype(int)
    n_artists = int(song['artist_name'].max() + 1)

    row_art = df_all['msno'].fillna(0).astype(int).values
    col_art = df_all['artist_name'].values

    # [FIX]: Đảm bảo shape là tuple of ints
    R_art = coo_matrix((data_ones, (row_art, col_art)), shape=(n_users, n_artists))

    u_art, s_art, vt_art = svds(R_art.astype(float), k=n_components_art)

    # Save Latent Factors Artist
    u_art_cols = [f'member_artist_component_{i}' for i in range(n_components_art)]
    mem_art_svd = pd.DataFrame(u_art, columns=u_art_cols)
    mem_art_svd['msno'] = range(n_users)
    member = member.merge(mem_art_svd, on='msno', how='left')

    # Merge Artist vectors vào Song
    art_cols = [f'artist_component_{i}' for i in range(n_components_art)]
    art_svd = pd.DataFrame(vt_art.T, columns=art_cols)
    art_svd['artist_name'] = range(n_artists)
    song = song.merge(art_svd, on='artist_name', how='left')

    # 4. Dot Products
    print("Calculating Dot Products...")

    def fast_dot(df, u_mat, s_val, v_mat, u_col, i_col):
        u_idx = df[u_col].fillna(0).astype(int).values
        i_idx = df[i_col].fillna(0).astype(int).values
        
        # Clip index an toàn
        i_idx = np.clip(i_idx, 0, v_mat.shape[0]-1)
        
        return np.sum((u_mat[u_idx] * v_mat[i_idx]) * s_val, axis=1)

    df_all['song_embeddings_dot'] = fast_dot(df_all, u_song, s_song, vt_song.T, 'msno', 'song_id')
    df_all['artist_embeddings_dot'] = fast_dot(df_all, u_art, s_art, vt_art.T, 'msno', 'artist_name')

    # Gán ngược lại Train/Test
    train_len = len(train)
    train['song_embeddings_dot'] = df_all['song_embeddings_dot'].values[:train_len]
    train['artist_embeddings_dot'] = df_all['artist_embeddings_dot'].values[:train_len]

    test['song_embeddings_dot'] = df_all['song_embeddings_dot'].values[train_len:]
    test['artist_embeddings_dot'] = df_all['artist_embeddings_dot'].values[train_len:]

    # 5. Save Final Files
    print("Saving final SVD processed files...")
    train.to_csv(os.path.join(TEMP_DIR, 'train_id_cnt_svd.csv'), index=False)
    test.to_csv(os.path.join(TEMP_DIR, 'test_id_cnt_svd.csv'), index=False)
    member.to_csv(os.path.join(TEMP_DIR, 'members_id_cnt_svd.csv'), index=False)
    song.to_csv(os.path.join(TEMP_DIR, 'songs_id_cnt_isrc_svd.csv'), index=False)

    print(">>> HOÀN THÀNH TOÀN BỘ LOGIC XỬ LÝ DỮ LIỆU.")

if __name__ == "__main__":
    main()
