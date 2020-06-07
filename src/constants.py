ON_KAGGLE = False

DATA_DIR = '/kaggle/input/input/global-wheat-detection' if ON_KAGGLE else "../../data"
TRAIN_DIR = f'{DATA_DIR}/train'
META_TRAIN = f'{DATA_DIR}/train.csv'
TEST_DIR = f'{DATA_DIR}/test'