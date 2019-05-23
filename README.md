rethinking-evs
==============================

Scripts of our CVPR'19 paper "Rethinking the Evaluation of Video Summaries" [[arXiv](https://arxiv.org/abs/1903.11328)]

# Setup
1. Create an environment.

`
$ conda env create -f environment.yml
`

2. Activate the new environment.

`
$ conda activate vsum_eval
`

# Data

### SumME
The data can be downloaded from the [project page](https://gyglim.github.io/me/vsum/index.html).
Copy the files in `GT/` to `data/raw/summe/GT/`.

### TVSum
Follow the steps described in the [TVSum Github page](https://github.com/yalesong/tvsum).
Copy `ydata-tvsum50.mat` to `data/raw/tvsum/`

Optional: For evaluate video summaries using KTS segmentation, we use KTS segmentation results provided [here](https://github.com/kezhang-cs/Video-Summarization-with-LSTM).
Download `shot_SumMe.mat` and `shot_TVSum.mat` and copy it to `data/raw/summe(or tvsum)/`.

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── interim
    │   ├── processed
    │   └── raw # please see "Data" description above
    │        ├── summe
    │        │     └── GT/ 
    │        │     └── shot_SumMe.mat
    │        ├── tvsum
    │        │     └── ydata-tvsum50.mat
    │        │     └── shot_TVSum.mat
    │        └── example.json
    ├── notebooks
    └── src

# Evaluate your video summaries on SumMe
We provide an evaluation script that also computes baseline scores with 100 trials.
For evaluating your own video summaries on SumMe, please use the following format and save the results in a JSON file.

```
{
    "video name":
    {
    'summary': [x1, x2, ... xn] # frame-level 0/1 labels
    'segment': [s1, s2, ... sm] # segmentation results
    },
    ...
}
```

xi=1 when i-th frame is in an output summary, otherwise 0.
s1, s2, ... sm are indices of frames corresponding to shot boundaries.
An example is in `data/raw/example.json`.
To evaluate the summarization results, run `src/summe_eval.py` as

```
python src/summe_eval.py path/to/json_file
```

The evaluation results will be saved to `data/processed/json_file.eval.csv`.
