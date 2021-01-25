from project1 import *
# 사이킷런 라이브러리는 ....
# 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공합니다.
# train_test_split은 난수 초기값을 지정할 수 있고,
# 행의 개수가 같은 여러 개의 데이터 셋을 넘겨 같은 인덱스를 기반으로 나눌 수 있습니다.
from sklearn.model_selection import train_test_split

fetch_housing_data()

housing = load_housing_data()
print(housing.head())
print(housing.info())
# 숫자형 특성의 요약정보를 보여줍니다.
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
plt.show()

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

housing["income_cat"].hist()

