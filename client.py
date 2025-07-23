import socket
import json
import pandas as pd
import joblib
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from together import Together
from sklearn.decomposition import PCA

HOST = 'localhost'
PORT = 9999
TOGETHER_API_KEY = "d2c99719a5a1b639417e45035e0fbf1ff3dfb668bb32cae0f8d25cbc10b9cba1"
LOG_FILE = 'anomalies.csv'
PLOT_FILE = 'network_traffic_plot.png'

model = joblib.load("anomaly_model.joblib")
llm_client = Together(api_key=TOGETHER_API_KEY)
history_df = pd.DataFrame()

def setup_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['timestamp', 'src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol', 'llm_confidence_score',
                 'llm_label', 'llm_reason'])

setup_log_file()

def log_anomaly(data, llm_response):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            data['src_port'],
            data['dst_port'],
            data['packet_size'],
            data['duration_ms'],
            data['protocol'],
            llm_response.get('confidence_score', 'N/A'),
            llm_response.get('label', 'N/A'),
            llm_response.get('reason', 'N/A')
        ])

def update_visualization(df):
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='duration_ms', y='packet_size', hue='anomaly', style='protocol', s=100,
                    palette={'Normal': 'skyblue', 'Anomaly': 'red'})
    plt.title('Network Traffic Analysis: Normal vs. Anomaly')
    plt.xlabel('Connection Duration (ms)')
    plt.ylabel('Packet Size (bytes)')
    plt.legend(title='Traffic Status')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOT_FILE)
    plt.close()


def plot_with_pca(df):
    df_encoded = pd.get_dummies(df.copy(), columns=["protocol"], drop_first=True)

    expected_cols = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_cols]

    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df_encoded)

        df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        df_plot['anomaly'] = df['anomaly'].values

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='anomaly',
                        palette={'Normal': 'skyblue', 'Anomaly': 'red'})
        plt.title("PCA of Network Traffic Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pca_plot.png")
        plt.close()
        print("📈 PCA plot saved to 'pca_plot.png'")
    except Exception as e:
        print(f"PCA Plot Error: {e}")



def pre_process_data(df):
    df_encoded = pd.get_dummies(df, columns=["protocol"], drop_first=True)

    expected_cols = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']

    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[expected_cols]
    return df_encoded


def get_llm_caption(data):
    if not llm_client:
        return {"label": "LLM_DISABLED", "reason": "API key not configured.", "confidence_score": 0.0}

    prompt = f"""
    Anomalous network traffic has been detected with the following characteristics:
    - Source Port: {data['src_port']}
    - Destination Port: {data['dst_port']}
    - Packet Size: {data['packet_size']} bytes
    - Duration: {data['duration_ms']} ms
    - Protocol: {data['protocol']}

    Based on this data, provide a concise JSON object with three keys:
    1. "label": A short, descriptive label for this anomaly (e.g., "Suspicious Port Activity", "Unusually Large Packet").
    2. "reason": A brief explanation of why this traffic is considered anomalous and a potential cause.
    3. "confidence_score": A numerical value from 0.0 (not confident) to 1.0 (very confident) indicating your confidence that this is a true anomaly.
    """
    messages = [{"role": "system",
                 "content": "You are a network security analyst. Your task is to analyze network data and provide clear, concise labels, explanations, and a confidence score for anomalies in a JSON format."},
                {"role": "user", "content": prompt}]

    try:
        response = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"label": "LLM_ERROR", "reason": str(e), "confidence_score": 0.0}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.connect((HOST, PORT))
    except ConnectionRefusedError:
        print(f"Connection Error: Connection to {HOST}:{PORT} was refused.")
        print("Please ensure the 'server.py' script is running.")
        exit()

    buffer = ""
    print(f"Client connected to server at {HOST}:{PORT}. Waiting for data...")
    print(f"Plots will be saved to '{PLOT_FILE}' as the data arrives.")

    while True:
        try:
            chunk = s.recv(1024).decode()
            if not chunk:
                print("\nServer closed the connection.")
                break
            buffer += chunk
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    data = json.loads(line)

                    df = pd.DataFrame([data])
                    processed_df = pre_process_data(df.copy())
                    prediction = model.predict(processed_df)
                    is_anomaly = prediction[0] == -1

                    temp_df = df.copy()
                    temp_df['anomaly'] = 'Anomaly' if is_anomaly else 'Normal'
                    history_df = pd.concat([history_df, temp_df], ignore_index=True)

                    if is_anomaly:
                        print("\n🚨 Anomaly Detected! Sending to LLM for analysis...")
                        llm_response = get_llm_caption(data)
                        print(f"Label: {llm_response.get('label', 'N/A')}")
                        print(f"Reason: {llm_response.get('reason', 'N/A')}")
                        print(f"Confidence Score: {llm_response.get('confidence_score', 'N/A')}\n")
                        log_anomaly(data, llm_response)
                    else:
                        print("Received normal data point.")

                    if len(history_df) % 5 == 0:
                        update_visualization(history_df)
                        plot_with_pca(history_df)
                        print(f"Plot updated and saved to '{PLOT_FILE}'")

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from received line: {line}")

        except KeyboardInterrupt:
            print("\nClient shutdown.")
            break
        except Exception as e:
            print(f"Critical error in main loop: {e}")
            break

print("Connection closed.")
