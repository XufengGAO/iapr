# README for IAPR Final Project

## Members of Group32

- [Ziyi ZHAO](https://github.com/Jacoo-Zhao)
- [Yujie HE](https://github.com/hibetterheyj)
- [Xufeng GAO](https://github.com/XufengGAO)

## Installation instructions

```shell
python -m pip install treys==0.1.4
python -m pip install imutils
# to install custom yolodetector
python -m pip install -e .
```

## Folder structure

```shell
❯ tree -L 1
.
├── card_utils.py # extraction utility functions
├── chip_utils.py # chip extraction utility functions
├── detect_utils.py # card detection utility functions
├── preprocess_utils.py # card preprocessing utility functions
├── utils.py # provides utility functions
├── viz_utils.py # visualization utility functions
├── data/ # data folder
├── yolodetector/ # developed yolo detector folder
├── setup.py # setup file for yolodetector package
├── project_complete_example.ipynb # notebook with full examples
├── project.ipynb # notebook for evaluating the project
├── test_process_image.py # test script for `process_image()`
└── readme.md
```

## Data

- data_project_poker.zip: <https://drive.google.com/file/d/1d7rOe88kEK1CEaLvYgNZkxrtKImLVC9X/view>

- constructed card dataset (Tcards.npy): <https://drive.google.com/file/d/1eVEAf7qZaTpfqygPjPsCIT5nNrsKcMk7/view>

```bash
# download the data from the google drive
cd project
gdown 1d7rOe88kEK1CEaLvYgNZkxrtKImLVC9X -O data_project_poker.zip
unzip data_project_poker.zip
gdown 1eVEAf7qZaTpfqygPjPsCIT5nNrsKcMk7
```

## Misc
### Announcements on the Moodle

- [Project announces and treys library Errors](https://moodle.epfl.ch/mod/forum/discuss.php?d=76543)
- Updated CSV: `./data/train/updated_train_labels.csv`
