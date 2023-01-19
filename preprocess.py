import pickle
import os
import sys
from tqdm import tqdm
import utils
from midi_processor_fixed.processor import encode_midi
from joblib import Parallel, delayed

def preprocess_midi(path):
    return encode_midi(path)

def process_one(path, save_dir):
    try:
        data = preprocess_midi(path)

        with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)
        return 0
    except:
        return 1

def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    # for path in tqdm(midi_paths):
    #     print(' ', end='[{}]'.format(path), flush=True)

    #     try:
    #         data = preprocess_midi(path)
        
    #         with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
    #             pickle.dump(data, f)
    #     except:
    #         pass
        
    skipped = Parallel(n_jobs=-1, prefer="processes")(delayed(process_one)(p, save_dir) for p in tqdm(midi_paths))
    print("Total: ", len(midi_paths), "skipped", sum(skipped))
if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
pass