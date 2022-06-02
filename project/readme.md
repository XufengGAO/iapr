# README for IAPR final project

## Announcements on the Moodle

- [Project announces and treys library Errors](https://moodle.epfl.ch/mod/forum/discuss.php?d=76543)
- Updated CSV: `./data/train/updated_train_labels.csv`

## Installation instructions

```shell
python -m pip install treys==0.1.4
# to install custom yolodetector
python -m pip install -e .
```

## Folder structure

:construction:

## Presentation

:construction:

## Data

<https://drive.google.com/file/d/1d7rOe88kEK1CEaLvYgNZkxrtKImLVC9X/view>

```bash
# download the data from the google drive
cd project
gdown 1d7rOe88kEK1CEaLvYgNZkxrtKImLVC9X -O data_project_poker.zip
unzip data_project_poker.zip
```

### constructed card dataset

- Tcards.npy: <https://drive.google.com/file/d/1eVEAf7qZaTpfqygPjPsCIT5nNrsKcMk7/view>

### structure

```shell
data
│
└─── image_setup
│    │    back_cards.jpg      # Back of the cards (either blue or red)
│    │    chips.jpg           # Set of chips used (red, green, blue, black, white)
│    │    kings.jpg           # Kings from the 4 colors (diamond, heart, spade, club)
│    │    spades_suits.jpg    # All cards of spades (2 - 10, Jack, Queen, King, Ace)
│    │    table.jpg           # Empty table
│    └─── ultimate_test.jpg   # If it works on that image, you would probably end up with a good score
│
└─── train
│    │    train_00.jpg        # Train image 00
│    │    ...
│    │    train_27.jpg        # Train image 27
│    └─── train_labels.csv    # Ground truth of the train set
│
└─── test
     │    test_00.jpg         # Test image 00 (day of the exam only)
     │    ...
     └─── test_xx             # Test image xx (day of the exam only)
```

## Misc

### Evaluation

#### Before the exam

- Create a zipped folder named **groupid_xx.zip** that you upload on moodle (xx being your group number).
- Include a **runnable** code (Jupyter Notebook and external files) and your presentation in the zip folder.

#### The day of the exam

- You will be given a new folder with few images, but no ground truth (csv file).
- We will ask you to run your pipeline in real time and to send us your prediction of the task you obtain with the provided function save_results.
- On our side, we will compute the performance of your classification algorithm.
- To evaluate your method, we will use the evaluate_game function presented below. To understand how the provided functions work, please read the documentation of the functions in utils.py.
- Please make sure your function returns the proper data format to avoid points penalty on the day of the exam.
