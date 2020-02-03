import numpy as np
import tensorflow as tf
import keras.backend as K
import argparse
import os
import mido
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
from tqdm import tqdm
from keras import losses
from collections import deque
from datasets.music_dataset import *
from constants.music_generation_constants import *
from datasets.midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def primary_loss(y_true, y_pred):
    # 3 separate loss calculations based on if note is played or not
    played = y_true[:, :, :, 0]
    bce_note = losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    bce_replay = losses.binary_crossentropy(y_true[:, :, :, 1], tf.multiply(played, y_pred[:, :, :, 1]) + tf.multiply(1 - played, y_true[:, :, :, 1]))
    mse = losses.mean_squared_error(y_true[:, :, :, 2], tf.multiply(played, y_pred[:, :, :, 2]) + tf.multiply(1 - played, y_true[:, :, :, 2]))
    return bce_note + bce_replay + mse

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def time_axis(dropout):
    def f(notes, beat, emotion):
        time_steps = int(notes.get_shape()[1])

        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)

        # Create features for every single note.
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes),
            Lambda(pitch_class_in_f(time_steps))(notes),
            Lambda(pitch_bins_f(time_steps))(notes),
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])

        x = note_features

        # [batch, notes, time, features]
        x = Permute((2, 1, 3))(x)

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate emotion
            emotion_proj = Dense(int(x.get_shape()[3]))(emotion)
            emotion_proj = TimeDistributed(RepeatVector(NUM_NOTES))(emotion_proj)
            emotion_proj = Activation('tanh')(emotion_proj)
            emotion_proj = Dropout(dropout)(emotion_proj)
            emotion_proj = Permute((2, 1, 3))(emotion_proj)
            x = Add()([x, emotion_proj])

            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)

        # [batch, time, notes, features]
        return Permute((2, 1, 3))(x)
    return f

def note_axis(dropout):
    dense_layer_cache = {}
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')

    def f(x, chosen, emotion):
        time_steps = int(x.get_shape()[1])
        
        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])))(chosen)
        
        x = Concatenate(axis=3)([x, shift_chosen])

        for l in range(NOTE_AXIS_LAYERS):
            # Integrate emotion
            if l not in dense_layer_cache:
                dense_layer_cache[l] = Dense(int(x.get_shape()[3]))

            emotion_proj = dense_layer_cache[l](emotion)
            emotion_proj = TimeDistributed(RepeatVector(NUM_NOTES))(emotion_proj)
            emotion_proj = Activation('tanh')(emotion_proj)
            emotion_proj = Dropout(dropout)(emotion_proj)
            x = Add()([x, emotion_proj])

            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)

            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)

        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    emotion_in = Input((time_steps, NUM_EMOTIONS))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    # Distributed representations
    emotion_l = Dense(EMOTION_UNITS, name='emotion')
    emotion = emotion_l(emotion_in)

    """ Time axis """
    time_out = time_axis(dropout)(notes, beat, emotion)

    """ Note Axis & Prediction Layer """
    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen, emotion)

    model = Model([notes_in, chosen_in, beat_in, emotion_in], [notes_out])
    model.compile(optimizer='nadam', loss=[primary_loss])

    """ Generation Models """
    time_model = Model([notes_in, beat_in, emotion_in], [time_out])

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    emotion_gen_in = Input((1, NUM_EMOTIONS), name='emotion_in')

    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    emotion_gen = emotion_l(emotion_gen_in)

    note_gen_out = naxis(note_features, chosen_gen, emotion_gen)

    note_model = Model([note_features, chosen_gen_in, emotion_gen_in], note_gen_out)

    return model, time_model, note_model

def build_or_load(allow_load=True):
    models = build_models()
    if allow_load:
        try:
            models[0].load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return models

def train(use_data_generator = False):
    models = build_or_load()
    print('Loading data')
    

    cbs = [
        ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='loss', patience=5)
    ]

    print('Training')
    if use_data_generator:
        generator = data_generator(EMOTIONS, BATCH_SIZE, SEQ_LEN)
        models[0].fit_generator(generator=generator,
            epochs=1000,
            verbose=1,
            callbacks=cbs,
            steps_per_epoch=STEPS_PER_EPOCH,
            workers=4)
    else:
        train_data, train_labels = load_all(EMOTIONS, SEQ_LEN)
        models[0].fit(train_data,
                      train_labels,
                      epochs=1000,
                      callbacks=cbs,
                      batch_size=BATCH_SIZE)
        
class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, emotion, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.emotion_memory = deque([emotion for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)

        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.emotion_memory)
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.emotion_memory)[-1:])
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # Apply volume
            self.next_note[n, 2] = vol
            # Flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, emotions):
    print('Generating with emotions:', emotions)

    _, time_model, note_model = models
    generations = [MusicGeneration(emotion) for emotion in emotions]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        
        mid = midi_encode(unclamp_midi(result))
        
        mid.save(fpath)
        
def main():
    parser = argparse.ArgumentParser(description='Movies analysis.')
    parser.add_argument('--emotion', default = 'Neutral', type=str, help='Emotion for music [Neutral, LALV, LAHV, HALV, HAHV]')
    parser.add_argument('--len', default = 10, type=int, help='Music length in seconds')
    parser.add_argument('--out', default = 'Neutral_example',type=str, help='Out file name')
    args = parser.parse_args()
    
    assert args.len > 0
    assert len(args.out) > 0
    try:
        emotion_id = EMOTIONS.index(args.emotion)
    except ValueError as val:
        print("No such emotion! Choose from: " + str(EMOTIONS))
        return
    emotion_hot = one_hot(emotion_id , NUM_EMOTIONS)

    seconds = args.len
    bars = math.ceil(seconds/BARS_PER_SECONDS)
    models = build_or_load()
    write_file(args.out, generate(models, bars, [emotion_hot]))
    
if __name__ == '__main__':
    main()
