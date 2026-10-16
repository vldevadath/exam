import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel("/home/KasiR/Downloads/Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

X = rfm[["Recency", "Frequency", "Monetary"]]

km = KMeans(n_clusters=3, random_state=1, n_init=10)
y = km.fit_predict(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

plot_tree(model, feature_names=X.columns, filled=True)
plt.show()
