import pandas as pd
import numpy as np
import gc
import os
from config import TEMP_DIR, WORK_DIR

def main():
    print("--- Đang chạy: preprocess_data_export.py ---")

    # ==============================================================================
    # PHẦN 1: XỬ LÝ TRAIN
    # ==============================================================================
    print("\n1. Đang xử lý Train Data...")
    try:
        train = pd.read_csv(os.path.join(TEMP_DIR, 'train_id_cnt_svd_stamp_before_after.csv'))
    except FileNotFoundError:
        print("Error: train_id_cnt_svd_stamp_before_after.csv not found. Run preprocess_before_after.py first.")
        return

    print(f"   Train shape: {train.shape}")

    # Lưu ngay ra file đích (Giữ nguyên float_format gốc)
    train.to_csv(os.path.join(WORK_DIR, 'train.csv'), index=False, float_format='%.6f')
    print("   -> Đã lưu train.csv")

    # Xóa ngay khỏi RAM
    del train
    gc.collect() # Dọn dẹp RAM ngay lập tức

    # ==============================================================================
    # PHẦN 2: XỬ LÝ TEST
    # ==============================================================================
    print("\n2. Đang xử lý Test Data...")
    try:
        test = pd.read_csv(os.path.join(TEMP_DIR, 'test_id_cnt_svd_stamp_before_after.csv'))
    except FileNotFoundError:
        print("Error: test_id_cnt_svd_stamp_before_after.csv not found.")
        return

    print(f"   Test shape: {test.shape}")

    test.to_csv(os.path.join(WORK_DIR, 'test.csv'), index=False, float_format='%.6f')
    print("   -> Đã lưu test.csv")

    del test
    gc.collect()

    # ==============================================================================
    # PHẦN 3: XỬ LÝ MEMBERS
    # ==============================================================================
    print("\n3. Đang xử lý Member Data...")
    try:
        member = pd.read_csv(os.path.join(TEMP_DIR, 'members_id_cnt_svd_stamp.csv'))
    except FileNotFoundError:
        print("Error: members_id_cnt_svd_stamp.csv not found.")
        return

    # --- Tạo Members cho GBDT ---
    print("   Creating members_gbdt.csv...")
    member.to_csv(os.path.join(WORK_DIR, 'members_gbdt.csv'), index=False)

    # --- Tạo Members cho NN (Neural Network) ---
    print("   Creating members_nn.csv...")
    member['bd_missing'] = np.isnan(member['bd'].values) * 1

    # FillNA theo logic gốc
    columns = ['bd']
    for col in columns:
        member[col] = member[col].fillna(np.nanmean(member[col]))

    # Xử lý timestamp std
    min_std = member['msno_timestamp_std'].min()
    if pd.isna(min_std): min_std = 0
    member['msno_timestamp_std'] = member['msno_timestamp_std'].fillna(min_std)

    member.to_csv(os.path.join(WORK_DIR, 'members_nn.csv'), index=False)
    print("   -> Đã lưu xong member files")

    del member
    gc.collect()

    # ==============================================================================
    # PHẦN 4: XỬ LÝ SONGS
    # ==============================================================================
    print("\n4. Đang xử lý Song Data...")
    try:
        song = pd.read_csv(os.path.join(TEMP_DIR, 'songs_id_cnt_isrc_svd_stamp.csv'))
    except FileNotFoundError:
        print("Error: songs_id_cnt_isrc_svd_stamp.csv not found.")
        return

    # --- Tạo Songs cho GBDT ---
    print("   Creating songs_gbdt.csv...")
    
    song_gbdt = song.copy()
    columns = ['composer', 'lyricist', 'language', 'first_genre_id', 'second_genre_id', 'third_genre_id']
    for col in columns:
        song_gbdt[col] = song_gbdt[col].fillna(0).astype(int)

    # Xử lý artist_name
    max_artist = song_gbdt['artist_name'].max()
    if pd.isna(max_artist): max_artist = 0
    song_gbdt['artist_name'] = song_gbdt['artist_name'].fillna(max_artist+1).astype(int)

    song_gbdt['isrc_missing'] = song_gbdt['isrc_missing'].astype(int)
    song_gbdt.to_csv(os.path.join(WORK_DIR, 'songs_gbdt.csv'), index=False)

    del song_gbdt
    gc.collect()

    # --- Tạo Songs cho NN ---
    print("   Creating songs_nn.csv...")
    # Quay lại với biến 'song' (dữ liệu chưa bị fill 0 ở bước trên)
    song['song_id_missing'] = np.isnan(song['song_length'].values) * 1

    columns = ['song_length', 'genre_id_cnt', 'artist_song_cnt', 'composer_song_cnt', \ 
           'lyricist_song_cnt', 'genre_song_cnt', 'song_rec_cnt', \ 
           'artist_rec_cnt', 'composer_rec_cnt', 'lyricist_rec_cnt', \ 
           'genre_rec_cnt', 'yy', 'cc_song_cnt', \ 
           'xxx_song_cnt', 'yy_song_cnt', 'cc_rec_cnt', 'xxx_rec_cnt', \ 
           'yy_rec_cnt', 'song_timestamp_std', 'artist_cnt', 'lyricist_cnt', \ 
           'composer_cnt', 'is_featured'] + ['artist_component_%d'%i for i in range(16)]

    for col in columns:
        if col in song.columns:
            mean_val = np.nanmean(song[col])
            if pd.isna(mean_val): mean_val = 0
            song[col] = song[col].fillna(mean_val)

    song.to_csv(os.path.join(WORK_DIR, 'songs_nn.csv'), index=False)
    print("   -> Đã lưu xong song files")

    del song
    gc.collect()

    print("\n--- HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ---")
    print("RAM đã được giải phóng. Bạn có thể xóa folder temporal_data ngay bây giờ.")

if __name__ == "__main__":
    main()
