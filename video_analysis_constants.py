import os
from enum import Enum

PATH_TO_PROJECT = os.getcwd()
PATH_TO_DATA = "/home/tomasz/Dokumenty/shared/"
DATASET_FOLDER = "video"
EXTRACTED_DATA_FOLDER = "extracted_data"
DATASET_RESULT_FILE = "ACCEDEranking.txt"
VIDO_EXTENSION = ".mp4"
PATH_TO_DATASET = os.path.join(PATH_TO_DATA, DATASET_FOLDER)
OUT_FOLDER = 'video_analysis/out'
MODEL_DIR = os.path.join(OUT_FOLDER, 'models')
MODEL_FILE = os.path.join(OUT_FOLDER, 'model.h5')
CACHE_DIR = os.path.join(OUT_FOLDER, 'cache')

MIN_NEUTRAL_LEVEL_VALUE = 3600
MAX_NEUTRAL_LEVEL_VALUE = 6300
SPLIT_LEVEL = 5000
FILTER_NUM = 700

FRAME_SIZE = 100
MAX_FRAMES = 300


PERCENT_OF_TRAIN = 0.8

class ValueLevel(Enum):
    Low = 1
    LowN = 2
    HighN = 3
    High = 4
    
class VideoClass(Enum):
    Neutral = 0
    LALV = 1
    LAHV = 2
    HALV = 3
    HAHV = 4