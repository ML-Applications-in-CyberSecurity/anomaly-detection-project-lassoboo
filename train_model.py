import json
import pandas as pd
import numpy as np
import random
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest

COMMON_SERVICES = {
    "HTTP": 80,
    "HTTPS": 443,
    "SSH": 22,
    "WEB_API": 8080
}


def generate_realistic_network_sample():
    service_name = random.choice(list(COMMON_SERVICES.keys()))
    src_port = COMMON_SERVICES[service_name]

    if service_name == "SSH":
        protocol = "TCP"
    else:
        protocol = random.choices(["TCP", "UDP"], weights=[0.8, 0.2])[0]

    hour = datetime.now().hour
    is_night = hour < 6 or hour > 22

    if protocol == "TCP":
        packet_size = int(random.gauss(1000, 180))
    else:
        packet_size = int(random.gauss(400, 100))

    packet_size = max(100, min(1500, packet_size))

    if is_night:
        duration_ms = int(random.gauss(250, 80))
    else:
        duration_ms = int(random.gauss(120, 40))

    duration_ms = max(50, min(500, duration_ms))

    if src_port in [80, 443, 8080]:
        dst_port = random.randint(1024, 65535)
    else:
        dst_port = random.randint(49152, 65535)

    return {
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_size": packet_size,
        "duration_ms": duration_ms,
        "protocol": protocol
    }


def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=["protocol"], drop_first=True)
    return df_encoded


if __name__ == "__main__":
    print("Generating synthetic training data...")
    dataset = [generate_realistic_network_sample() for _ in range(1000)]

    with open("training_data.json", "w") as f:
        json.dump(dataset, f, indent=2)

    with open("training_data.json") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    preprocessed_df = preprocess_data(df)

    print(f"Preprocessed shape: {preprocessed_df.shape}")

    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    print("Columns used for training:", preprocessed_df.columns.tolist())

    model.fit(preprocessed_df.values)
    print("🧪 Final columns after preprocessing:")
    print(preprocessed_df.columns.tolist())

    joblib.dump(model, "anomaly_model.joblib")
    print("Model saved as 'anomaly_model.joblib'")

    sample_predictions = model.predict(preprocessed_df.values[:10])
    print("Sample predictions:", sample_predictions)

    anomaly_scores = model.decision_function(preprocessed_df.values)
    print("Average anomaly score:", np.mean(anomaly_scores))
