from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']
print(cities)


encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
print(city_labels)


encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
print(encoder.fit_transform(city_labels))