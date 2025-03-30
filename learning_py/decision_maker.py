import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class MethodSelector:

    def __init__(self, clf=None):
        self.clf = clf

    def load_model(self, model_path="my_method_selector.pkl"):
        self.clf = joblib.load(model_path)

    def save_model(self, model_path="my_method_selector.pkl"):
        joblib.dump(self.clf, model_path)

    def train(self, X_train, y_train, n_estimators=100, max_depth=5, random_state=42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.clf.fit(X_train, y_train)

    def predict_method(self, features):
        arr = np.array(features).reshape(1, -1)
        prediction = self.clf.predict(arr)
        return int(prediction[0])

def main():
    filename = "decision_maker_training_data.txt"  # replace with your actual file name
    lines = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.strip("[] \t")
            lines.append(line)

    X, y = [], []
    for line in lines:
        parts = [p.strip() for p in line.split()]
        if len(parts) != 3:
            continue

        float_parts = []
        for p in parts:
            if p.lower() == "inf":
                float_parts.append(1e15)
            else:
                float_parts.append(float(p))

        runtime1, runtime2, similarity_score = float_parts

        label = 1 if runtime1 < runtime2 else 0

        X.append([similarity_score])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Total rows parsed:", len(X))
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

  
    selector = MethodSelector()
    selector.train(X_train, y_train, n_estimators=100, max_depth=5, random_state=42)


    y_pred = selector.clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy on test set: {:.3f}".format(acc))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))

 
    selector.save_model("my_method_selector.pkl")
    print("Model saved to 'my_method_selector.pkl'")

if __name__ == "__main__":
    main()

