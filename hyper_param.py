ip_file_dir = "./data/"
CLASS_NUM = category_num = 200
max_len = 110
BATCH_SIZE = 256
LEARNING_RATE = 0.01
epochs = 50


def get_train_length():
    import pandas as pd
    train_file = f"{ip_file_dir}{category_num}/train.csv"
    df = pd.read_csv(train_file)
    return len(df.index)


N = get_train_length()
SUBSET_SIZE = 0.1
WORKERS = 1