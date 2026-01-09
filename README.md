# Recsys-KKBOX-competition-Late-

Code for our attempt in KKBOX Music Recommendation competition.

## Structure

The code is organized in `src/` directory.

### Preprocessing

The preprocessing pipeline consists of several steps that must be run in order.

1.  **ID Processing**: Encodes User and Song IDs.
    ```bash
    python src/preprocess_ids.py
    ```
2.  **Counts & SVD**: Generates count features, ISRC features, and SVD components.
    ```bash
    python src/preprocess_counts.py
    ```
3.  **Timestamp Processing**: Processes timestamp windows (computationally intensive).
    ```bash
    python src/preprocess_timestamp.py
    ```
4.  **Before/After Sequence**: Generates sequential features (prev/next song).
    ```bash
    python src/preprocess_before_after.py
    ```
5.  **Data Export**: Finalizes and exports `train.csv`, `test.csv`, and auxiliary files for training.
    ```bash
    python src/preprocess_data_export.py
    ```

### Training

After preprocessing (Step 5 completed), you can train any model individually.

*   **LightGBM** (Top 1 Solution Logic):
    ```bash
    python src/train_lightgbm.py
    ```
    Output: `submission_lightgbm.csv`

*   **XGBoost**:
    ```bash
    python src/train_xgboost.py
    ```
    Output: `submission_xgboost.csv`

*   **CatBoost**:
    ```bash
    python src/train_catboost.py
    ```
    Output: `submission_catboost.csv`

*   **DeepFM** (PyTorch):
    ```bash
    python src/train_deepfm.py
    ```
    Output: `submission_deepfm.csv`

## Requirements

*   pandas
*   numpy
*   scikit-learn
*   scipy
*   lightgbm
*   xgboost
*   catboost
*   torch (for DeepFM)

## Notes

*   Ensure the raw data files (`train.csv`, `test.csv`, `members.csv`, `songs.csv`, `song_extra_info.csv`) are located in the project root directory.
*   Temporary files are stored in `temporal_data/`.