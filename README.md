# Aligned Music Transformer

TLDR; Use Reinforcement Learning From Human Feedback (RLHF) to tune a Music Transformer to generate Piano music that I like. 

Examples of pieces I love:
1. [Somnus - Dreaming of the Dawn](https://www.youtube.com/watch?v=PgWa9HoZMU0)
2. [Sorrow Without Solace](https://www.youtube.com/watch?v=35sjWiTKCKc)
3. [Secret of the Forest](https://www.youtube.com/watch?v=3vgTnT5iKQc)


## Data
Combine [ADL-Piano-MIDI](https://github.com/lucasnfe/adl-piano-midi), [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano); and MIDI files scraped from [Ichigo's](https://ichigos.com/sheets/) and [VGMusic](http://www.vgmusic.com/music/other/miscellaneous/piano/). Scripts to scrape from Ichigo's and VGMusic are under [here](./scrape/).

The combined dataset is about 22K MIDI files, totalling about 1800 hours of piano music. To the best of my knowledge, this results in the biggest publicly available Piano MIDI dataset (the second biggest being GiantMIDI, 1200 hours).

### Method 1
First download [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano) and [ADL-Piano-MIDI](https://github.com/lucasnfe/adl-piano-midi), and  extract under `data/giant_midi` and `data/adl-piano-midi` respectively. Then run the following:

```bash
python3 scrape/ichigos.py
python3 scrape/vgmusic.py
cp data/vgmusic/* data/all/
cp data/ichigos/* data/all/
cp data/giant_midi/* data/all/
cp data/adl-piano-midi/midi/adl-piano-midi/*/*/*/*.mid data/all/
```
### Method 2
Download from this link (link to be added later).

### Preprocessing

`python3 preprocess.py data/all data/processed`

Changes: Parallelized the existing MIDI preprocessing code, and replaced `progress` with `tqdm`