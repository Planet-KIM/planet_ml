import os
import tarfile
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
import pandas as pd

DOWNLOAD_PATH = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_PATH + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    
    #
    os.makedirs(housing_path, exist_ok=True)
    #
    tgz_path = os.path.join(housing_path, "housing.tgz")
    #
    urlretrieve(housing_url, tgz_path)
    #
    housing_tgz = tarfile.open(tgz_path)
    #
    housing_tgz.extractall(path=housing_path)
    #
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):

    #
    csv_path = os.path.join(housing_path, "housing.csv")
    #
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


# 각 샘플마다 식별자의 해시 값을 계산하여 해시 최댓 값의 20% 작거나
# 같은 샘플만 테스트 세트로 보낼 수 있습니다.
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))

    return data.loc[~in_test_set], data.loc[in_test_set]
