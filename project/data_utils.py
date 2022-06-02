import pandas as pd


def getGameDict(game_id=0, label_fn='data/train/updated_train_labels.csv'):
    game_labels = pd.read_csv('data/train/updated_train_labels.csv')
    game_labels = game_labels.fillna('0')
    game = game_labels.iloc[[game_id]]
    game_dict = {key: game.get(key).values[0] for key in game.columns}
    return game_dict
