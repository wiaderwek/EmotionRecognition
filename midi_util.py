"""
Handles MIDI file loading
"""
import mido
import numpy as np
import os
from constants import *

def midi_encode(note_seq, resolution=NOTES_PER_BEAT, step=120):
    """
    Takes a piano roll and encodes it into MIDI file
    """
    # Instantiate a MIDI File (contains a list of tracks)
    midi = mido.MidiFile()
    midi.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI messages)
    track = mido.MidiTrack()
    # Append the track to the midi
    midi.tracks.append(track)

    play = note_seq[:, :, 0]
    replay = note_seq[:, :, 1]
    volume = note_seq[:, :, 2]

    # The current pattern being played
    current = np.zeros_like(play[0])
    # Absolute tick of last event
    last_msg_tick = 0

    for tick, data in enumerate(play):
        data = np.array(data)

        if not np.array_equal(current, data):# or np.any(replay[tick]):

            for index, next_volume in np.ndenumerate(data):
                if next_volume > 0 and current[index] == 0:
                    # Was off, but now turned on
                    msg = mido.Message(NOTE_ON,
                                       note = index[0],
                                       velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                                       time = (tick - last_msg_tick) * step
                                      )
                    track.append(msg)
                    last_msg_tick = tick
                elif current[index] > 0 and next_volume == 0:
                    # Was on, but now turned off
                    msg = mido.Message(NOTE_OFF,
                                       note = index[0],
                                       time = (tick - last_msg_tick) * step
                                      )
                    track.append(msg)
                    last_msg_tick = tick

                elif current[index] > 0 and next_volume > 0 and replay[tick][index[0]] > 0:
                    # Handle replay
                    msg_off = mido.Message(NOTE_OFF,
                                       note = index[0],
                                       time = (tick - last_msg_tick) * step
                                      )
                    track.append(msg_off)
                    msg_on = mido.Message(NOTE_ON,
                                       note = index[0],
                                       velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                                       time = 0
                                      )
                    track.append(msg_on)
                    last_msg_tick = tick

        current = data

    tick += 1

    # Turn off all remaining on notes
    for index, vol in np.ndenumerate(current):
        if vol > 0:
            # Was on, but now turned off
            msg = mido.Message(NOTE_OFF,
                                   note = index[0],
                                   time = (tick - last_msg_tick) * step
                                  )
            track.append(msg)
            last_msg_tick = tick

    # Add the end of track event, append it to the track
    #eot = mido.Message(END_OF_TRACK,
    #                   time = 0
    #                  )
    #track.append(eot)

    return midi

def midi_decode(midi,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI file and decodes it into a piano roll.
    """
    if step is None:
        step = midi.ticks_per_beat // NOTES_PER_BEAT

    # Extract all tracks at highest resolution
    merged_replay = None
    merged_volume = None

    for track in midi.tracks:
        # The downsampled sequences
        replay_sequence = []
        volume_sequence = []

        # Raw sequences
        replay_buffer = [np.zeros((classes,))]
        volume_buffer = [np.zeros((classes,))]

        for i, msg in enumerate(track):
            # Duplicate the last note pattern to wait for next event
            for _ in range(msg.time):
                replay_buffer.append(np.zeros(classes))
                volume_buffer.append(np.copy(volume_buffer[-1]))

                # Buffer & downscale sequence
                if len(volume_buffer) > step:
                    # Take the min
                    replay_any = np.minimum(np.sum(replay_buffer[:-1], axis=0), 1)
                    replay_sequence.append(replay_any)

                    # Determine volume by max
                    volume_sum = np.amax(volume_buffer[:-1], axis=0)
                    volume_sequence.append(volume_sum)

                    # Keep the last one (discard things in the middle)
                    replay_buffer = replay_buffer[-1:]
                    volume_buffer = volume_buffer[-1:]

            if msg.type == END_OF_TRACK:
                break

            # Modify the last note pattern
            if msg.type == NOTE_ON:
                velocity = msg.velocity
                note = msg.note
                volume_buffer[-1][note] = velocity / MAX_VELOCITY

                # Check for replay_buffer, which is true if the current note was previously played and needs to be replayed
                if len(volume_buffer) > 1 and volume_buffer[-2][note] > 0 and volume_buffer[-1][note] > 0:
                    replay_buffer[-1][note] = 1
                    # Override current volume with previous volume
                    volume_buffer[-1][note] = volume_buffer[-2][note]

            if msg.type == NOTE_OFF:
                velocity = msg.velocity
                note = msg.note
                volume_buffer[-1][note] = 0

        # Add the remaining
        replay_any = np.minimum(np.sum(replay_buffer, axis=0), 1)
        replay_sequence.append(replay_any)
        volume_sequence.append(volume_buffer[0])

        replay_sequence = np.array(replay_sequence)
        volume_sequence = np.array(volume_sequence)
        assert len(volume_sequence) == len(replay_sequence)

        if merged_volume is None:
            merged_replay = replay_sequence
            merged_volume = volume_sequence
        else:
            # Merge into a single track, padding with zeros of needed
            if len(volume_sequence) > len(merged_volume):
                # Swap variables such that merged_notes is always at least
                # as large as play_sequence
                tmp = replay_sequence
                replay_sequence = merged_replay
                merged_replay = tmp

                tmp = volume_sequence
                volume_sequence = merged_volume
                merged_volume = tmp

            assert len(merged_volume) >= len(volume_sequence)

            diff = len(merged_volume) - len(volume_sequence)
            merged_replay += np.pad(replay_sequence, ((0, diff), (0, 0)), 'constant')
            merged_volume += np.pad(volume_sequence, ((0, diff), (0, 0)), 'constant')

    merged = np.stack([np.ceil(merged_volume), merged_replay, merged_volume], axis=2)
    # Prevent stacking duplicate notes to exceed one.
    merged = np.minimum(merged, 1)
    return merged

def load_midi(fname):
    try:
        midi = mido.MidiFile(fname)
    except OSError as e:
        print(fname)
    except ValueError as val:
        print(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    try:
        note_seq = np.load(cache_path)
    except Exception as e:
        # Perform caching
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        note_seq = midi_decode(midi)
        np.save(cache_path, note_seq)

    assert len(note_seq.shape) == 3, note_seq.shape
    assert note_seq.shape[1] == MIDI_MAX_NOTES, note_seq.shape
    assert note_seq.shape[2] == 3, note_seq.shape
    assert (note_seq >= 0).all()
    assert (note_seq <= 1).all()
    return note_seq

def clean_midi(fname):
    """
    Remove duplicated tracks 
    """
    midi = MidiFile(fname)

    message_numbers = []
    duplicates = []

    for track in midi.tracks:
        if len(track) in message_numbers:
            duplicates.append(track)
        else:
            message_numbers.append(len(track))

    for track in duplicates:
        midi.tracks.remove(track)

    midi.save(fname)

if __name__ == '__main__':
    # Test
    # p = midi.read_midifile("out/test_in.mid")
    p = midi.read_midifile("out/test_in.mid")
    p = midi_encode(midi_decode(p))
    midi.write_midifile("out/test_out.mid", p)