import os

# Define the musical emotions
EMOTIONS = [
    "Neutral",
    "LALV",
    "LAHV",
    "HALV",
    "HAHV"
]

NUM_EMOTIONS = len(EMOTIONS)

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 8
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 20
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Notes
NOTE_ON = 'note_on'
NOTE_OFF = 'note_off'
END_OF_TRACK = 'end_of_track'
PROGRAM_CHANGE = 'program_change'

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR
STEPS_PER_EPOCH = 10

# Hyper Parameters
OCTAVE_UNITS = 64
EMOTION_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
DATA_DIR = 'music'
OUT_DIR = 'music_generation/out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')