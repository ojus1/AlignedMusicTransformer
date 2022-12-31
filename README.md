# Aligned Music Transformer

TLDR; Use Reinforcement Learning From Human Feedback (RLHF) to tune a Music Transformer to generate Piano music that I like. 

Examples of pieces I love:
1. [Somnus - Dreaming of the Dawn](https://www.youtube.com/watch?v=PgWa9HoZMU0)
2. [Sorrow Without Solace](https://www.youtube.com/watch?v=35sjWiTKCKc)
3. [Secret of the Forest](https://www.youtube.com/watch?v=3vgTnT5iKQc)


# Data
Combine [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano); and MIDI files scraped from [Ichigo's](https://ichigos.com/sheets/) and [VGMusic](http://www.vgmusic.com/music/other/miscellaneous/piano/).

Scripts to scrape from Ichigo's and VGMusic are under [here](./scrape/)
First download [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano) and extract under `data/giant_midi`.

```bash
python3 scrape/ichigos.py
python3 scrape/vgmusic.py
ln -s data/vgmusic/* data/all/
ln -s data/ichigos/* data/all/
ln -s data/giant_midi/* data/all/
```

# Preprocessing

`python3 preprocess.py data/all data/processed`

Changes: Parallelized the existing MIDI preprocessing code, and replaced `progress` with `tqdm`