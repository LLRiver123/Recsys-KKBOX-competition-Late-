import os

# Get the absolute path of the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define directories
INPUT_DIR = os.path.join(PROJECT_ROOT, 'kkbox') # Assuming data is in 'kkbox' folder or root
# Note: User provided file list shows data might be in root or 'kkbox (2).ipynb' suggests it's just there.
# Let's assume input data (csvs) are in the project root for now, or the user will place them there.
# If the user has a specific data folder, they should update this.
# Based on the notebook, it used /kaggle/input/kkbox... 
# I will set INPUT_DIR to PROJECT_ROOT for simplicity as standard CSVs usually sit there in these tasks.
INPUT_DIR = PROJECT_ROOT 

WORK_DIR = PROJECT_ROOT
TEMP_DIR = os.path.join(WORK_DIR, 'temporal_data')

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    print(f"Created temporary directory: {TEMP_DIR}")

print(f"Project Root: {PROJECT_ROOT}")
print(f"Temp Directory: {TEMP_DIR}")
