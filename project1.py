import os
import tarfile
from urllib.request import urlretrieve

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

fetch_housing_data()

import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):

    #
    csv_path = os.path.join(housing_path, "housing.csv")
    #
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head())
print(housing.info())
# 숫자형 특성의 요약정보를 보여줍니다.
print(housing.describe())

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
plt.show()
