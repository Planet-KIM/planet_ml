from project1 import *
# 사이킷런 라이브러리는 ....
# 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공합니다.
# train_test_split은 난수 초기값을 지정할 수 있고,
# 행의 개수가 같은 여러 개의 데이터 셋을 넘겨 같은 인덱스를 기반으로 나눌 수 있습니다.
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def test_Housing(plot_housing=True):

    fetch_housing_data()
    housing = load_housing_data()
    
    print(housing.head())
    print(housing.info())

    # 숫자형 특성의 요약정보를 보여줍니다.
    print(housing.describe())

    if plot_housing is True:
        housing.hist(bins=50, figsize=(20,15))
        plt.show()
    else:
        pass

    return housing

housing = custom_Housing(plot_housing=False)


def split_housing(plot_housing=True):
    
    housing = load_housing_data()
    train_set, test_set = split_train_test(housing, 0.2)

    print(len(train_set))
    print(len(test_set))

    # 'index' 열이 추가된 데이터프레임이 반환됩니다.
    housing_with_id = housing.reset_index()

    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

    # 위도와 경도는 안정적인 데이터임으로 이를 이용해서
    # 고유적인 데이터를 생성합니다.
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # 카테고리 5개를 가진 소득 카테고리 특성을 만듭니다.
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    if plot_housing is True:
        housing["income_cat"].hist()
        plt.show()
    else:
        pass

    return housing

housing = split_housing(plot_housing=False)

def stratified_Housing(plot_housing=True):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    if plot_housing is True:
        # 훈련세트를 손상하지 않게 하기위해서 복사
        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude")
        plt.show()

        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
        plt.show()

        # 인구 밀집도와 가격의 연관성을 그래프로 출력하기
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
        plt.legend()
        plt.show()

    else:
        pass

    return housing


def corr_housing():

print("\n+ 표준 상관계수 구하기 +")
# 표준 상관계수 구하기
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# 숫장형 특성 사이에 산점도를 그래주는 메소드
#scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()

#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()

# 필요한 특성 조합 만들기.
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

###### Ready to make datasets ######
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 특성에 값이 없을 때 이를 처리해주는 방법
housing.dropna(subset=["total_bedrooms"]) # 방법 1 - (해당 구역을 제거합니다.)
housing.drop("total_bedrooms", axis=1) # 방법 2 - (전체 특성을 삭제합니다.)
median = housing["total_bedrooms"].median() # 방법 3 (누락된 값을 채우는 방법)
housing["total_bedrooms"].fillna(median, inplace=True)

# 누락된 값을 손쉽게 다루게 해주는 메소드
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
# 중간 값을 계산해서 그 결과를 밑의 함수에 저장
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)
# 변형된 특성들이 들어있는 Numpy Array
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr)

# 텍스트 특성
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder =  OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
