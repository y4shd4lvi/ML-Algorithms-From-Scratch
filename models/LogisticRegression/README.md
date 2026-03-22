ML-Algorithms-From-Scratch
A modular, interactive Streamlit application that implements and visually compares classic machine learning algorithms built from the ground up — side-by-side with their scikit-learn counterparts on live synthetic datasets.

Overview
This project is designed for learning and experimentation. Each algorithm is implemented from scratch in plain NumPy so you can see exactly what's happening under the hood — no black boxes. A polished Streamlit UI lets you tune hyperparameters, step through training epoch-by-epoch, and visually compare decision boundaries in real time.

Repository Structure
ML-Algorithms-From-Scratch/
│
├── models/
│   └── LogisticRegression/
│       ├── app.py               # Streamlit UI — sidebar controls, layout, state
│       ├── data_utils.py        # Dataset generation & preprocessing
│       ├── models.py            # Perceptron (scratch) + sklearn Logistic Regression
│       ├── visualization.py     # All matplotlib figure factories
│       └── requirements.txt     # Python dependencies
│
└── README.md

Algorithms Covered
🟡 Perceptron — implemented from scratch
The classic Rosenblatt (1958) binary classifier, built entirely in NumPy.
ComponentDetailActivationStep function — output ∈ {−1, +1}Update rulew ← w + η · y · x on misclassificationBias updateb ← b + η · yConvergenceGuaranteed on linearly separable dataTrackingSlope, intercept, and accuracy saved per epoch
🟢 Logistic Regression — scikit-learn
A probabilistic linear classifier used as the reference model.
ComponentDetailOutputP(y = 1 | x) via sigmoid σ(z) = 1 / (1 + e^−z)LossBinary cross-entropyOptimiserL-BFGS (sklearn default)AdvantageHandles non-separable data; stable probabilities

Features

Interactive epoch slider — replay the Perceptron's boundary evolution step-by-step; all prior boundaries shown as faint dashed traces
Side-by-side boundary plots — final decision regions of both models with accuracy badges
Training error curve — misclassification count across epochs highlighted up to the current slider position
Live metric cards — train/test accuracy for both models, sample count, and final-epoch error count
Sidebar controls — adjust everything without touching code


Getting Started
1. Clone the repository
bashgit clone https://github.com/<your-username>/ML-Algorithms-From-Scratch.git
cd ML-Algorithms-From-Scratch/models/LogisticRegression
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app.py
The app will open at http://localhost:8501.

Sidebar Controls
ControlDescriptionRangeNumber of samplesTotal dataset size100 – 1 000Class separationHow distinct the two classes are0.3 – 3.0Random seedReproducibility seed0 – 9 999Training epochsPerceptron training iterations5 – 200Learning ratePerceptron step size η0.001 – 1.0

Module Responsibilities
data_utils.py
Generates a 2-feature, 2-class synthetic dataset via sklearn.make_classification, applies StandardScaler, and performs a stratified train/test split. Returns a single dict consumed by the rest of the app — keeping data logic fully isolated.
models.py
Contains two independent components:

Perceptron — full from-scratch implementation. fit() runs weight updates per sample, records a (slope, intercept, accuracy) snapshot after every epoch, and stores per-epoch error counts.
train_logistic_regression() — thin wrapper around sklearn.LogisticRegression for consistent API.
logistic_boundary_params() — extracts slope and intercept from a fitted LR model so both algorithms share the same plotting interface.

visualization.py
Pure figure factories — no Streamlit imports. Each function accepts data arrays and trained model objects and returns a matplotlib.Figure. Functions: plot_dataset, plot_final_boundaries, plot_perceptron_epoch, plot_accuracy_comparison. All figures share a consistent dark theme via shared palette constants.
app.py
Streamlit entry point. Manages session state so models are only retrained when parameters actually change. Renders hero header, metric cards, all plot sections, epoch slider, expandable algorithm explainers, and footer.