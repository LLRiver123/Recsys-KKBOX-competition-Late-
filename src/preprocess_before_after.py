import numpy as np
import pandas as pd
import os
from collections import defaultdict
from config import TEMP_DIR

def main():
    print("--- Đang chạy: preprocess_before_after.py (FIXED) ---")

    try:
        tr = pd.read_csv(os.path.join(TEMP_DIR, 'train_id_cnt_svd_stamp.csv'))
        te = pd.read_csv(os.path.join(TEMP_DIR, 'test_id_cnt_svd_stamp.csv'))
    except FileNotFoundError:
        print("Error: '_stamp.csv' files not found. Run preprocess_timestamp.py first.")
        return

    print('data loaded.')

    ## continous index
    # FIX: append -> concat
    concat = pd.concat([tr[['msno', 'song_id', 'source_type', 'source_screen_name', 'timestamp']],
                        te[['msno', 'song_id', 'source_type', 'source_screen_name', 'timestamp']]])

    ## before data
    song_dict = defaultdict(lambda: None)
    type_dict = defaultdict(lambda: None)
    name_dict = defaultdict(lambda: None)
    time_dict = defaultdict(lambda: None)

    before_data = np.zeros((len(concat), 4))
    concat_values = concat[['song_id', 'source_type', 'source_screen_name', 'timestamp']].values
    concat_msno = concat['msno'].values

    # Loop xuôi (Before)
    print("Processing Before Data...")
    for i in range(len(concat)):
        msno = concat_msno[i]

        if(song_dict[msno] == None):
            before_data[i] = concat_values[i]
            before_data[i, 3] = np.nan
        else:
            before_data[i, 0] = song_dict[msno]
            before_data[i, 1] = type_dict[msno]
            before_data[i, 2] = name_dict[msno]
            before_data[i, 3] = time_dict[msno]

        song_dict[msno] = concat_values[i, 0]
        type_dict[msno] = concat_values[i, 1]
        name_dict[msno] = concat_values[i, 2]
        time_dict[msno] = concat_values[i, 3]

    print('data before done.')

    ## after data
    song_dict = defaultdict(lambda: None)
    type_dict = defaultdict(lambda: None)
    name_dict = defaultdict(lambda: None)
    time_dict = defaultdict(lambda: None)

    after_data = np.zeros((len(concat), 4))

    # Loop ngược (After)
    print("Processing After Data...")
    for i in range(len(concat))[::-1]:
        msno = concat_msno[i]

        if(song_dict[msno] == None):
            after_data[i] = concat_values[i]
            after_data[i, 3] = np.nan
        else:
            after_data[i, 0] = song_dict[msno]
            after_data[i, 1] = type_dict[msno]
            after_data[i, 2] = name_dict[msno]
            after_data[i, 3] = time_dict[msno]

        song_dict[msno] = concat_values[i, 0]
        type_dict[msno] = concat_values[i, 1]
        name_dict[msno] = concat_values[i, 2]
        time_dict[msno] = concat_values[i, 3]

    print('data after done.')

    ## to_csv
    idx = 0
    for i in ['song_id', 'source_type', 'source_screen_name', 'timestamp']:
        tr['before_'+i] = before_data[:len(tr), idx]
        tr['after_'+i] = after_data[:len(tr), idx]

        te['before_'+i] = before_data[len(tr):, idx]
        te['after_'+i] = after_data[len(tr):, idx]

        idx += 1

    for i in ['song_id', 'source_type', 'source_screen_name']:
        tr['before_'+i] = tr['before_'+i].fillna(-1).astype(int)
        te['before_'+i] = te['before_'+i].fillna(-1).astype(int)
        tr['after_'+i] = tr['after_'+i].fillna(-1).astype(int)
        te['after_'+i] = te['after_'+i].fillna(-1).astype(int)

    # === SỬA LỖI Ở ĐÂY ===

    # 1. Tính toán khoảng cách thời gian và xử lý số âm/NaN TRƯỚC khi log
    # fillna(0) tạm thời cho phép trừ để tránh lỗi, sau đó clip(lower=0) để đảm bảo không âm
    tr['before_timestamp'] = (tr['timestamp'] - tr['before_timestamp']).fillna(0).clip(lower=0)
    te['before_timestamp'] = (te['timestamp'] - te['before_timestamp']).fillna(0).clip(lower=0)

    tr['before_timestamp'] = np.log1p(tr['before_timestamp'])
    te['before_timestamp'] = np.log1p(te['before_timestamp'])

    tr['after_timestamp'] = (tr['after_timestamp'] - tr['timestamp']).fillna(0).clip(lower=0)
    te['after_timestamp'] = (te['after_timestamp'] - te['timestamp']).fillna(0).clip(lower=0)

    tr['after_timestamp'] = np.log1p(tr['after_timestamp'])
    te['after_timestamp'] = np.log1p(te['after_timestamp'])

    # 2. Thay thế fillna(inplace=True) bằng phép gán
    # Vì ở bước trên ta đã fillna(0) để tính toán rồi, nên thực tế bước này có thể dư thừa
    # nhưng giữ lại logic mean của tác giả cho chắc chắn (nếu có logic khác)
    mean_tr_before = np.nanmean(tr['before_timestamp'])
    mean_te_before = np.nanmean(te['before_timestamp'])
    mean_tr_after = np.nanmean(tr['after_timestamp'])
    mean_te_after = np.nanmean(te['after_timestamp'])

    tr['before_timestamp'] = tr['before_timestamp'].fillna(mean_tr_before)
    te['before_timestamp'] = te['before_timestamp'].fillna(mean_te_before)
    tr['after_timestamp'] = tr['after_timestamp'].fillna(mean_tr_after)
    te['after_timestamp'] = te['after_timestamp'].fillna(mean_te_after)

    # =====================

    tr.to_csv(os.path.join(TEMP_DIR, 'train_id_cnt_svd_stamp_before_after.csv'), index=False)
    te.to_csv(os.path.join(TEMP_DIR, 'test_id_cnt_svd_stamp_before_after.csv'), index=False)

    print("Hoàn thành preprocess_before_after.py")

if __name__ == "__main__":
    main()
