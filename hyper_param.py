ip_file_dir = "./data/"
CLASS_NUM = category_num = 350
max_len = 110
BATCH_SIZE = 128
LEARNING_RATE = 0.001
epochs = 2


def get_train_length():
    import pandas as pd
    train_file = f"{ip_file_dir}{category_num}/train.csv"
    df = pd.read_csv(train_file)
    return len(df.index)


N = get_train_length()
SUBSET_SIZE = 0.1
WORKERS = 4