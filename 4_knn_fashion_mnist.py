import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("/home/KasiR/Downloads/archive/fashion-mnist_train.csv")
test_data = pd.read_csv("/home/KasiR/Downloads/archive/fashion-mnist_test.csv")

x_train = train_data.iloc[:5000, 1:] / 255.0
y_train = train_data.iloc[:5000, 0]

x_test = test_data.iloc[:1000, 1:] / 255.0
y_test = test_data.iloc[:1000, 0]

for k in [3, 7, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print("K : ", k, "Accuracy : ", round(accuracy, 2))
