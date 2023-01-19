from midi_processor_fixed import processor
import random
import numpy as np

def transpose_timestretch(notes_idx, shift, stretch):
    assert(isinstance(notes_idx, np.ndarray))
    assert(notes_idx.ndim == 1)

    midi_data = processor.decode_midi(notes_idx)
    for instrument in midi_data.instruments:
        # Don't want to shift drum notes
        if not instrument.is_drum:
            for note in instrument.notes:
                # pitch transpose
                note.pitch += shift

                # time stretch
                duration = note.end - note.start 
                center = (duration) / 2
                stretched_duration = duration * stretch
                note.start = max(0., center - stretched_duration / 2)
                note.end = center + stretched_duration / 2

    return processor.encode_midi(mid=midi_data)

def pad_sequences(sequences, pad_token, direction, max_len):
    new_seqs = []
    for s in sequences:
        s = np.array(s, dtype=np.int32)
        if len(s) < max_len:
            if direction == "right":
                s = np.concatenate([s, np.full(max_len - len(s), pad_token, dtype=s.dtype)])
            elif direction == "left":
                s = np.concatenate([np.full(max_len - len(s), pad_token, dtype=s.dtype), s])
        new_seqs.append(s.reshape(1, -1))
    return np.concatenate(new_seqs, axis=0)

def augment_batch(x_batch, y_batch, eos_token, sos_token, shift_range=3):
    # x_batch = x_batch.numpy()
    # y_batch = y_batch.numpy()

    assert(x_batch.ndim == 2)
    assert(y_batch.ndim == 2)
    x_new = []
    y_new = []
    max_len = x_batch.shape[1]

    for i in range(x_batch.shape[0]):
        mag = random.choice(list(range(shift_range)))
        sign = random.choice([-1, 1])
        shift = mag * sign

        stretch = random.choice([0.95, 0.975, 1.0, 1.025, 1.05])

        x_new.append(transpose_timestretch(x_batch[i], shift, stretch))
        y_new.append(transpose_timestretch(y_batch[i], shift, stretch))
    x_new = pad_sequences(x_new, sos_token, "left", max_len)
    y_new = pad_sequences(y_new, eos_token, "right", max_len)
    return x_new, y_new