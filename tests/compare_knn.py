from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from proxiss import ProxiKNN


def knn(X, y, k):
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(Xt, yt)
    preds = model.predict(Xv)
    report = classification_report(yv, preds)
    print(report)


def proxi_flat(X, y, k):
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = ProxiKNN(k, 12, "l2")
    model.fit(Xt, yt)
    preds = model.predict_batch(Xv)
    report = classification_report(yv, preds)
    print(report)


X, y = make_classification(n_classes=2, n_samples=20000, n_features=200)
knn(X, y, 5)
proxi_flat(X, y, 5)
