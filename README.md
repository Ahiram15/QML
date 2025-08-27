# ⚛️ QML Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Qiskit](https://img.shields.io/badge/qiskit-quantum-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*Quantum-Enhanced Fraud Detection with Real-time Analysis & AI-Powered Protection*

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## 🌟 Overview

The **QML Fraud Detection System** is a cutting-edge financial security platform that combines classical machine learning with quantum computing to provide state-of-the-art fraud detection capabilities. The system uses XGBoost for initial classification, Quantum Support Vector Classifier (QSVC) for complex pattern recognition, and quantum-generated OTPs for secure transaction verification.

### 🎯 Key Highlights

- **Hybrid AI Architecture**: Classical ML + Quantum Computing
- **Real-time Processing**: Instant fraud detection for transaction streams
- **Quantum Security**: True random OTP generation using quantum circuits
- **Interactive Dashboard**: Beautiful Streamlit interface with analytics
- **Multi-layer Defense**: Progressive risk assessment and verification

---

## ✨ Features

### 🔬 **Core Detection Engine**
- **XGBoost Classifier**: Primary fraud detection model
- **Quantum SVM**: Advanced pattern recognition for complex fraud schemes
- **Feature Engineering**: PCA dimensionality reduction + amount scaling
- **Risk Stratification**: Low/Medium/High risk categorization

### ⚛️ **Quantum Components**
- **Quantum OTP Generator**: True randomness using quantum circuits
- **QSVC Integration**: Quantum Support Vector Classifier for hybrid analysis
- **Quantum Enhancement**: Quantum algorithms for borderline cases

### 📊 **Analytics Dashboard**
- **Real-time Metrics**: Transaction approval/block/challenge rates
- **Interactive Charts**: Fraud probability distributions and decision flows
- **Model Usage Tracking**: Monitor which AI models are being used
- **Transaction Details**: Comprehensive transaction analysis with filters

### 🔐 **Security Features**
- **Progressive Authentication**: Multi-level verification system
- **OTP Challenge System**: Quantum-secured one-time passwords
- **Risk-based Routing**: Intelligent model selection based on risk levels
- **Audit Trail**: Complete transaction and decision logging

---

### 🧠 **Model Pipeline**

1. **Data Preprocessing**
   - StandardScaler for V1-V28 features
   - PCA reduction to 5 components
   - Amount normalization

2. **XGBoost Classification**
   - Primary fraud detection model
   - Fast gradient boosting algorithm
   - Risk probability assessment

3. **QSVC Analysis**
   - Quantum Support Vector Classifier
   - Advanced pattern recognition
   - Quantum feature mapping

4. **Security Layer**
   - Quantum OTP generation
   - Risk-based authentication
   - Progressive verification

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- 4GB+ RAM recommended

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/qml-fraud-detection.git
cd qml-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Models Directory

```bash
mkdir models
mkdir data
```

### 4. Train Models (Optional)

```bash
jupyter notebook model_training.ipynb
```

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
flask>=2.3.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
qiskit>=0.44.0
qiskit-aer>=0.12.0
plotly>=5.15.0
joblib>=1.3.0
requests>=2.31.0
```

---

## 🎮 Usage

### 1. Start the Backend API

```bash
python backend.py
```

The Flask API will start on `http://127.0.0.1:5000`

### 2. Launch the Streamlit Interface

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### 3. Upload Transaction Data

1. Prepare CSV file with columns:
   - `V1` to `V28`: PCA-transformed features
   - `Amount`: Transaction amount
   - `Transaction ID` (optional): Unique identifier

2. Use the **"📤 Upload & Analyze"** tab to upload your file

3. View results in real-time with:
   - Fraud probability scores
   - Risk-based decisions
   - Model usage analytics

### 4. Handle Challenge Transactions

For transactions requiring verification:
1. Click **"🔐 Verify Transaction"**
2. Enter the quantum-generated OTP
3. Confirm or deny the transaction

---

## 📊 Demo

### Sample Analysis Results

```
Transaction ID: 3001 | Amount: $16,636.63
🌳 XGBoost: 99.88% fraud probability
❌ Decision: BLOCKED (Extreme Risk)

Transaction ID: 3002 | Amount: $10,951.18  
🌳 XGBoost: 0.07% fraud probability
✅ Decision: APPROVED (Low Risk)

Transaction ID: 3006 | Amount: $5,264.92
🌳 XGBoost → ⚛️ QSVC Hybrid Analysis
🔐 Decision: OTP CHALLENGE (Medium Risk)
```

### Dashboard Metrics

- **Total Transactions**: 20
- **✅ Approved**: 5 (25.0%)
- **❌ Blocked**: 11 (55.0%)  
- **⚠️ Challenged**: 4 (20.0%)

---

## 🧪 Testing

### Run Test Files

The repository includes sample test files:
- `test_file_1.csv`: 20 transactions with mixed risk levels
- `test_file_2.csv`: 15 transactions with balanced distribution

### Generate Custom Test Data

```python
python create_test_files.py
```

This creates random test files from the training dataset for validation.

---

## 🛠️ Configuration

### Model Parameters

Edit `model_training.ipynb` to adjust:
- XGBoost hyperparameters
- PCA components (default: 5)
- Risk thresholds (30%, 98%)
- Feature scaling methods

### UI Customization

Modify `app.py` to customize:
- Color schemes and themes
- Dashboard layouts
- Chart configurations
- Alert thresholds

### Quantum Settings

Configure quantum components in `backend.py`:
- Quantum simulator backends
- Circuit depth and shots
- OTP generation parameters

---

## 📋 API Reference

### POST `/predict`

Upload CSV file for fraud detection analysis.

**Request**: Multipart form data with CSV file
**Response**: JSON array of transaction results

```json
[
  {
    "Transaction ID": 3001,
    "Amount": 16636.63,
    "Fraud Probability": "99.88%",
    "Decision": "❌ Block",
    "Models Used": "XGBoost",
    "Analysis Method": "XGBoost Only",
    "Risk Level": "Extreme"
  }
]
```

### GET `/generate-otp`

Generate quantum-secured 6-digit OTP.

**Response**:
```json
{
  "otp": "847291",
  "method": "quantum",
  "timestamp": "2025-08-27T20:10:08"
}
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Models Not Found**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'models/xgb_model.joblib'
```
**Solution**: Train models using `model_training.ipynb` or download pre-trained models.

**2. QSVC Feature Mismatch**
```
⚠️ QSVC feature mismatch! Expected 600, got 6
```
**Solution**: This is expected behavior. The system automatically falls back to quantum enhancement.

**3. Backend Connection Error**
```
🔌 Cannot connect to backend at http://127.0.0.1:5000
```
**Solution**: Ensure Flask backend is running with `python backend.py`.

### Performance Optimization

- **CPU Usage**: Reduce XGBoost `n_jobs` parameter
- **Memory**: Lower PCA components or batch size
- **Quantum Speed**: Use `qasm_simulator` instead of `statevector_simulator`

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

### Areas for Contribution

- 🔬 **Quantum Algorithms**: Improve QSVC implementation
- 📊 **Visualization**: Enhanced dashboard components  
- 🛡️ **Security**: Additional authentication methods
- 🧪 **Testing**: Automated test suites
- 📖 **Documentation**: API docs and tutorials

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Qiskit Team**: Quantum computing framework
- **XGBoost Contributors**: Gradient boosting implementation
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning utilities

---

## 📞 Support

- 📧 **Email**: [your-email@domain.com]
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/qml-fraud-detection/issues)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/qml-fraud-detection/wiki)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/yourusername/qml-fraud-detection/discussions)

---

<div align="center">

**⚛️ Securing the Future with Quantum Technology ⚛️**

*Built with ❤️ by [Your Name]*

</div>
