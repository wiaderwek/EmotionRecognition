import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
import glob
import random
import functools
from keras import backend
from tqdm import tqdm
from enum import Enum
from subprocess import call
from copy import copy
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
import threading
from video_analysis_constants import *
      

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    @functools.wraps(func)
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen
    

class DataSet():
    def __init__(self, seq_length=40, image_shape=(224, 224, 3)):
        
        """Constructor.
        seq_length = (int) the number of frames to consider
        """
        self.path_to_dataset = PATH_TO_DATASET
        self.path_to_extracted_data = os.path.join(PATH_TO_DATA, EXTRACTED_DATA_FOLDER)
        self.seq_length = seq_length
        self.max_frames = MAX_FRAMES  # max number of frames a video can have for us to use it
        # Get the data.
        self.data = self.get_data()
        
        # Filter data
        #self.filter_number = self.find_divide_number()
        self.filter_number = FILTER_NUM
        self.data = self.filter_data(self.filter_number)
        
        #extract frames from videos
        self.extract_data()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape
        
    def get_data(self):
        data = []
        path_to_result = os.path.join(PATH_TO_DATA, DATASET_RESULT_FILE)
        list_of_results = open(path_to_result).readlines()
        list_of_results.pop(0)
        for res in list_of_results:
            splitted_res = res.split("\t")
            video_name = splitted_res[1]
            video_name = video_name[:-4]
            valency = splitted_res[2]
            arousal = splitted_res[3]
            emotion_class = DataSet.get_class_for_arousal_and_valency(int(arousal), int(valency))
            #number_of_frames = self.extract_data_for_video(video_name)
            #print(video_name + ": " + arousal + ", " + valency + ", " + emotion_class.name + ", " + str(number_of_frames))
            data.append([video_name, int(valency), int(arousal), emotion_class])

        return data
    
    def extract_data(self):
        for video in self.data:
            number_of_frames = self.extract_data_for_video(video[0])
            video.append(number_of_frames)
            
        
    @staticmethod
    def get_class_for_arousal_and_valency(arousal, valency):
        if arousal < MIN_NEUTRAL_LEVEL_VALUE:
            if valency < SPLIT_LEVEL:
                return VideoClass.LALV
            else:
                return VideoClass.LAHV
        elif arousal < SPLIT_LEVEL:
            if valency < MIN_NEUTRAL_LEVEL_VALUE:
                return VideoClass.LALV
            elif valency < MAX_NEUTRAL_LEVEL_VALUE:
                return VideoClass.Neutral
            else:
                return VideoClass.LAHV
        elif arousal < MAX_NEUTRAL_LEVEL_VALUE:
            if valency < MIN_NEUTRAL_LEVEL_VALUE:
                return VideoClass.HALV
            elif valency < MAX_NEUTRAL_LEVEL_VALUE:
                return VideoClass.Neutral
            else:
                return VideoClass.HAHV
        else:
            if valency < SPLIT_LEVEL:
                return VideoClass.HALV
            else:
                return VideoClass.HAHV
            
    def extract_data_for_video(self, video_name):
        if not os.path.isdir(self.path_to_extracted_data):
            os.mkdir(self.path_to_extracted_data)
        if not os.path.isdir(os.path.join(self.path_to_extracted_data, video_name)):
            os.mkdir(os.path.join(self.path_to_extracted_data, video_name))
            
        if not self.check_if_video_is_extracted(video_name):
            src = os.path.join(self.path_to_dataset, video_name + VIDO_EXTENSION)
            dest = os.path.join(self.path_to_extracted_data, video_name,
                        '%04d.jpg')
            call(["ffmpeg", "-i", src, dest])
        return len(self.get_frames_for_video(video_name))
            
    def get_frames_for_video(self, video_name):
        images = sorted(glob.glob(os.path.join(self.path_to_extracted_data, video_name, '*jpg')))
        return images
        
    def check_if_video_is_extracted(self, video_name):
        return bool(os.path.exists(os.path.join(self.path_to_extracted_data, video_name,
                               '0001.jpg')))
        
    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[4]) >= self.seq_length and int(item[4]) <= self.max_frames:
                data_clean.append(item)

        return data_clean
    
    def filter_data(self, num):
        """Limit samples to fewer then num for each VideoClass"""
        new_data = []
        neutral, lalv, lahv, halv, hahv = 0, 0, 0, 0, 0
        for row in self.data:
            if row[3] == VideoClass.Neutral and neutral < num:
                neutral += 1
                new_data.append(row)
            elif row[3] == VideoClass.LALV and lalv < num:
                lalv += 1
                new_data.append(row)
            elif row[3] == VideoClass.LAHV and lahv < num:
                lahv += 1
                new_data.append(row)
            elif row[3] == VideoClass.HALV and halv < num:
                halv += 1
                new_data.append(row)
            elif row[3] == VideoClass.HAHV and hahv < num:
                hahv += 1
                new_data.append(row)
        return new_data
    
    def find_divide_number(self):
        """Find minimum number of samples for each VideoClass"""
        neutral, lalv, lahv, halv, hahv = 0, 0, 0, 0, 0
        for row in self.data:
            if row[3] == VideoClass.Neutral:
                neutral += 1
            elif row[3] == VideoClass.LALV:
                lalv += 1
            elif row[3] == VideoClass.LAHV:
                lahv += 1
            elif row[3] == VideoClass.HALV:
                halv += 1
            elif row[3] == VideoClass.HAHV:
                hahv += 1
        return min(neutral, lalv, lahv, halv, hahv)
            
    
    def split_train_test(self, percent_of_train):
        number_of_train = int(len(self.data) * percent_of_train)
        y = copy(self.data)
        random.shuffle(y)
        train = y[:number_of_train]
        test = y[number_of_train:]
        
        return train, test
    
    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [self.process_image(x, self.image_shape) for x in frames]
    
    def process_image(self, image, target_shape):
        """Given an image, process it and return the array."""
        # Load the image.
        h, w, _ = target_shape
        image = load_img(image, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)

        return x
    
    def get_all_sequences_in_memory(self, precent_of_train):
        train, test = self.split_train_test(precent_of_train)
        
        X, y = self.load_data(train)
        X_test, y_test = self.load_data(test)
        return X, y, X_test, y_test
    
    def load_data(self, data):
        X, y = [], []
        for row in data:
            print(row[0])
            frames = self.get_frames_for_video(row[0])
            frames = DataSet.rescale_list(frames, self.seq_length)
            sequence = self.build_image_sequence(frames)
            X.append(sequence)
            y.append(self.get_class_one_hot(row[3]))
            
        return np.array(X), np.array(y)
            
    def get_generators(self, batch_size, precent_of_train):
        train, test = self.split_train_test(precent_of_train)
        
        return self.creata_generator(batch_size, train), self.creata_generator(batch_size, test)
    
    @threadsafe_generator
    def creata_generator(self, batch_size, data):
        while 1:
            X, y = [], []
            for _ in range(batch_size):
                sample = random.choice(data)
                frames = self.get_frames_for_video(sample[0])
                frames = DataSet.rescale_list(frames, self.seq_length)
                sequence = self.build_image_sequence(frames)
                X.append(sequence)
                y.append(self.get_class_one_hot(sample[3]))
        
            yield np.array(X), np.array(y)
            
    def get_class_one_hot(self, video_class):
        # Now one-hot it.
        label_hot = to_categorical(video_class.value, len(VideoClass))
        return label_hot
            
    @staticmethod        
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]