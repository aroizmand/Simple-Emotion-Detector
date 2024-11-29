import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATA_FILE = 'data.txt'

def train_and_save_model(data_file, model_file='model.pkl', scaler_file='scaler.pkl'):
    data = np.loadtxt(data_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=['HAPPY', 'SAD', 'SURPRISED']))

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {model_file}, Scaler saved to {scaler_file}")

if __name__ == "__main__":
    train_and_save_model(DATA_FILE)
