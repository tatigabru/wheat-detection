ON_KAGGLE = True

DIR_INPUT = '/kaggle/input' if ON_KAGGLE else "../data"
DIR_TRAIN = f'{DIR_INPUT}/global-wheat-detection/train'
META_TRAIN = f'{DIR_INPUT}/global-wheat-detection/train.csv'
DIR_TEST = f'{DIR_INPUT}/global-wheat-detection/test'

IMG_SIZE = 224
