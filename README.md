💤 Project Somnus — Sleep Stage Classifier (EEG-Based)

Somnus is a lightweight EEG-based sleep stage classification project built using open-source PhysioNet Sleep-EDF data.
It processes raw EEG signals, extracts meaningful frequency-domain features, and classifies sleep stages such as Wake (W), N1, N2, N3, and REM (R) using a machine learning pipeline.

🧠 Overview

Somnus simulates a real-world neuroscience workflow:

EEG Preprocessing — Bandpass & adaptive filtering (0.3–40 Hz).

Feature Extraction — Computes spectral power in delta, theta, alpha, sigma, and beta bands.

Classification — Uses a Random Forest model for stage prediction.

Evaluation — Outputs confusion matrix and class distribution visualizations.

⚙️ Tech Stack
Layer	Tools
Programming	Python (Scikit-learn, NumPy, Pandas)
EEG Toolkit	MNE-Python
Visualization	Matplotlib, Seaborn
Dataset	PhysioNet Sleep-EDF (EEG Polysomnography)
📊 Results (Sample Run)
Somnus accuracy: 31.43%  |  Balanced Accuracy: 22.54%

Metric	Score
Precision	0.29
Recall	0.23
F1-score	0.23

(Results vary across runs depending on EEG subject and data length.)

🧩 Visual Outputs
Output	Description
🧠 class_distribution.png	Sleep stage frequency distribution
📈 confusion_matrix.png	Model performance across stages

Both are saved automatically inside /assets/ after training.

🧰 How to Run Locally
# Clone repo
git clone https://github.com/devansh-29-glitch/Somnus.git
cd Somnus

# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download dataset
python somnus_download.py

# Train model
python somnus_train.py

💡 Future Enhancements

🧬 Add CNN/LSTM deep learning for improved accuracy

🕹 Real-time EEG visualization dashboard

💤 Personalized sleep scoring reports

👨‍💻 Author

Devansh Sharma
Machine Learning × Neuroscience Enthusiast


💬 “Dreams are just data — Somnus decodes them.”