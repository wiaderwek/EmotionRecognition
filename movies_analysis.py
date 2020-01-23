from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input, Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from collections import deque
import sys
import time
import os.path
from video_analysis_constants import *
from video_dataset import *

class AVAnalysisModel():   
    def __init__(self, seq_length, load_model=False):
        self.seq_length = seq_length
        self.input_shape = (seq_length, 80, 80, 3)
        
        self.model = self.lrcn()
        
        if load_model:
            self.load_model()
        else:
            #compile ony if don't load model to prevent losing optimizer states
            optimizer = Adam(lr=1e-5, decay=1e-6)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

    
    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=l2(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=l2(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        #model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        #model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 92, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 196, init=initialiser, reg_lambda=reg_lambda)
        
        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(len(VideoClass), activation='softmax'))

        return model
    
    def load_model(self):
        self.model.load_weights(MODEL_FILE)
        try:
            self.model.load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
            
def train(seq_length, load_model=False, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(OUT_FOLDER, 'logs'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)
    
    #Helper: Save the model during training
    checkpointer = ModelCheckpoint(
        filepath=MODEL_FILE,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)

    if image_shape is None:
        data = DataSet(
            seq_length=seq_length
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by PERCENT_OF_TRAIN to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * PERCENT_OF_TRAIN) // batch_size

    if load_to_memory:
        # Get data.
        X, y, X_test, y_test = data.get_all_sequences_in_memory(PERCENT_OF_TRAIN)
    else:
        # Get generators.
        generator, val_generator = data.get_generators(batch_size, PERCENT_OF_TRAIN)

    # Get the model.
    rm = AVAnalysisModel(seq_length, load_model)
    

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)
        
class EmotionAnalyser():
    def __init__(self, load_model=True):
        seq_length = 40
        
        self.model = AVAnalysisModel(seq_length, load_model)
        
    def predict(self, seq_length, image_shape, video_dir, video_name):
        # Get the data and process it.
        if image_shape is None:
            data = DataSet(seq_length=seq_length)
        else:
            data = DataSet(seq_length=seq_length, image_shape=image_shape)
        
        # Extract the sample from the data.
        sample = data.get_frames_to_predict(video_dir, video_name)
        
        # Predict!
        prediction = self.model.model.predict(np.expand_dims(sample, axis=0))
        print(prediction)
        print(np.argmax(prediction[0]))
        return np.argmax(prediction[0])