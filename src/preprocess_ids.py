import numpy as np
import pandas as pd
import os
import time
import gc
from sklearn.preprocessing import LabelEncoder
from config import INPUT_DIR, TEMP_DIR

# ==============================================================================
# PHẦN 1: ID PROCESS (Xử lý ID, encode và lưu file tạm)
# ==============================================================================
def main():
    print("--- Đang chạy: preprocess_ids.py ---")

    # 1. Load Data
    print("Loading raw data...")
    # Checking for existence of files to avoid errors if filename differs slightly
    try:
        members = pd.read_csv(os.path.join(INPUT_DIR, 'members.csv'))
        songs = pd.read_csv(os.path.join(INPUT_DIR, 'songs.csv'))
        songs_extra = pd.read_csv(os.path.join(INPUT_DIR, 'song_extra_info.csv'))
        train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
        test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure members.csv, songs.csv, song_extra_info.csv, train.csv, and test.csv are in the project root.")
        return

    # 2. Filter (Giữ lại bài hát/user xuất hiện trong train/test)
    song_id_set = set(pd.concat([train['song_id'], test['song_id']]))
    songs = songs[songs['song_id'].isin(song_id_set)].copy()
    songs_extra = songs_extra[songs_extra['song_id'].isin(song_id_set)].copy()

    msno_set = set(pd.concat([train['msno'], test['msno']]))
    members = members[members['msno'].isin(msno_set)].copy()

    print('Data loaded and filtered.')

    # 3. Preprocess MSNO (User ID)
    print('Encoding MSNO...')
    msno_encoder = LabelEncoder()
    all_msno = pd.concat([members['msno'], train['msno'], test['msno']]).unique()
    msno_encoder.fit(all_msno.astype(str))

    members['msno'] = msno_encoder.transform(members['msno'].astype(str))
    train['msno'] = msno_encoder.transform(train['msno'].astype(str))
    test['msno'] = msno_encoder.transform(test['msno'].astype(str))

    # 4. Preprocess Song ID
    print('Encoding Song ID...')
    song_id_encoder = LabelEncoder()
    all_songs = pd.concat([songs['song_id'], songs_extra['song_id'], train['song_id'], test['song_id']]).unique()
    song_id_encoder.fit(all_songs.astype(str))

    songs['song_id'] = song_id_encoder.transform(songs['song_id'].astype(str))
    songs_extra['song_id'] = song_id_encoder.transform(songs_extra['song_id'].astype(str))
    train['song_id'] = song_id_encoder.transform(train['song_id'].astype(str))
    test['song_id'] = song_id_encoder.transform(test['song_id'].astype(str))

    # 5. Preprocess Train/Test Features (Source info)
    print('Processing Source Info...')
    columns = ['source_system_tab', 'source_screen_name', 'source_type']
    for column in columns:
        column_encoder = LabelEncoder()
        combined = pd.concat([train[column], test[column]]).astype(str)
        column_encoder.fit(combined)
        train[column] = column_encoder.transform(train[column].astype(str))
        test[column] = column_encoder.transform(test[column].astype(str))

    # 6. Preprocess Members (City, Gender, Date)
    print('Processing Members Info...')
    columns = ['city', 'gender', 'registered_via']
    for column in columns:
        column_encoder = LabelEncoder()
        members[column] = members[column].fillna('Unknown').astype(str)
        column_encoder.fit(members[column])
        members[column] = column_encoder.transform(members[column])

    # Xử lý ngày tháng (Fix lỗi mktime với NaN)
    def safe_date_convert(x):
        try:
            return time.mktime(time.strptime(str(int(float(x))), '%Y%m%d'))
        except:
            return np.nan

    members['registration_init_time'] = members['registration_init_time'].apply(safe_date_convert)
    members['expiration_date'] = members['expiration_date'].apply(safe_date_convert)

    # 7. Preprocess Songs (Genre, Artist, Composer...)
    print('Processing Songs Info...')
    # Logic tách Genre cũ
    genre_id = np.zeros((len(songs), 4))
    song_genre_ids = songs['genre_ids'].fillna('0').astype(str).values

    for i in range(len(songs)):
        ids = song_genre_ids[i].split('|')
        l = len(ids)
        if l > 0: genre_id[i, 0] = int(ids[0])
        if l > 1: genre_id[i, 1] = int(ids[1])
        if l > 2: genre_id[i, 2] = int(ids[2])
        genre_id[i, 3] = l

    songs['first_genre_id'] = genre_id[:, 0]
    songs['second_genre_id'] = genre_id[:, 1]
    songs['third_genre_id'] = genre_id[:, 2]
    songs['genre_id_cnt'] = genre_id[:, 3]

    # Encode Genres
    genre_encoder = LabelEncoder()
    all_genres = pd.concat([songs['first_genre_id'], songs['second_genre_id'], songs['third_genre_id']]).unique()
    genre_encoder.fit(all_genres.astype(int))
    songs['first_genre_id'] = genre_encoder.transform(songs['first_genre_id'].astype(int))
    songs['second_genre_id'] = genre_encoder.transform(songs['second_genre_id'].astype(int))
    songs['third_genre_id'] = genre_encoder.transform(songs['third_genre_id'].astype(int))
    songs.drop('genre_ids', axis=1, inplace=True)

    # Helper functions for text counting
    def artist_count(x):
        x = str(x)
        return x.count('and') + x.count(',') + x.count(' feat') + x.count('&') + 1

    def get_count(x):
        try:
            x = str(x)
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        except:
            return 0

    def get_first_term(x):
        try:
            x = str(x)
            if x.count('|') > 0: x = x.split('|')[0]
            if x.count('/') > 0: x = x.split('/')[0]
            if x.count('\\') > 0: x = x.split('\\')[0]
            if x.count(';') > 0: x = x.split(';')[0]
            return x.strip()
        except:
            return x

    songs['artist_cnt'] = songs['artist_name'].apply(artist_count).astype(np.int8)
    songs['lyricist_cnt'] = songs['lyricist'].apply(get_count).astype(np.int8)
    songs['composer_cnt'] = songs['composer'].apply(get_count).astype(np.int8)
    songs['is_featured'] = songs['artist_name'].apply(lambda x: 1 if ' feat' in str(x) else 0).astype(np.int8)

    # Encode Text Columns
    songs['artist_name'] = songs['artist_name'].astype(str).apply(get_first_term)
    songs['lyricist'] = songs['lyricist'].astype(str).apply(get_first_term)
    songs['composer'] = songs['composer'].astype(str).apply(get_first_term)
    songs['language'] = songs['language'].fillna(-1).astype(str)

    for col in ['artist_name', 'lyricist', 'composer', 'language']:
        le = LabelEncoder()
        songs[col] = le.fit_transform(songs[col])

    # --- LƯU FILE ID (Bắt buộc phải lưu trước khi sang bước 2) ---
    print('Saving ID Processed files...')
    members.to_csv(os.path.join(TEMP_DIR, 'members_id.csv'), index=False)
    songs.to_csv(os.path.join(TEMP_DIR, 'songs_id.csv'), index=False)
    songs_extra.to_csv(os.path.join(TEMP_DIR, 'songs_extra_id.csv'), index=False)
    train.to_csv(os.path.join(TEMP_DIR, 'train_id.csv'), index=False)
    test.to_csv(os.path.join(TEMP_DIR, 'test_id.csv'), index=False)

    # Clean up
    del members, songs, songs_extra, train, test
    gc.collect()
    print("Preprocess IDs Completed.")

if __name__ == "__main__":
    main()
