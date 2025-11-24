# Customer Churn Prediction System

A comprehensive machine learning system that predicts customer churn in the telecom industry with an interactive Streamlit dashboard. This project features real-time predictions, model interpretability using SHAP, and actionable retention recommendations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Interactive Dashboard**: Beautiful blue and white themed interface
- **Multiple ML Models**: Choose between RandomForest and XGBoost
- **Real-Time Predictions**: Predict churn probability for individual customers
- **Model Interpretability**: SHAP values for understanding feature importance
- **Business Insights**: Actionable recommendations for retention strategies
- **Performance Metrics**: Comprehensive model evaluation with ROC curves and confusion matrices

## Project Structure

```
customer-churn/
│
├── Data/
│   └── telecom_churn.csv          # Dataset file
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
└── model.pkl                       # Trained model (generated after training)
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

Open your terminal/command prompt and run:

```bash
git clone https://github.com/HayaShahid/customer-churn.git
cd customer-churn
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Your Data

1. Ensure your dataset `telecom_churn.csv` is in the `Data` folder
2. The CSV should contain these columns:
   - State
   - Account length
   - Area code
   - International plan
   - Voice mail plan
   - Number vmail messages
   - Total day minutes
   - Total day calls
   - Total day charge
   - Total eve minutes
   - Total eve calls
   - Total eve charge
   - Total night minutes
   - Total night calls
   - Total night charge
   - Total intl minutes
   - Total intl calls
   - Total intl charge
   - Customer service calls
   - Churn

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

##  Using the Dashboard

### 1. Dashboard Page
- View overall churn statistics
- Explore data visualizations
- Understand churn patterns

### 2. Model Training Page
- Select between RandomForest or XGBoost
- Train your model with one click
- View performance metrics and ROC curves

### 3. Prediction Page
- Enter customer information
- Get real-time churn predictions
- View churn probability scores

### 4. Model Insights Page
- Analyze feature importance
- Understand SHAP values
- Interpret model decisions

### 5. Recommendations Page
- View retention strategies
- Get actionable insights
- Implement business recommendations

##  Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web dashboard framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **SHAP**: Model interpretability
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Additional plotting

##  Model Performance

The system achieves:
- **Accuracy**: ~85-90%
- **Precision**: ~80-85%
- **Recall**: ~75-80%
- **F1-Score**: ~77-82%

##  Key Insights

1. **Customer Service Calls**: Customers with 4+ service calls are at high risk
2. **International Plan**: Higher churn rate among international plan users
3. **Day Charges**: High day charges correlate with increased churn
4. **Account Length**: Newer accounts show higher churn probability

##  Uploading to GitHub from VS Code

### Method 1: Using VS Code Interface

1. **Initialize Git Repository**:
   - Open VS Code
   - Open Terminal (`Ctrl + ` ` or View → Terminal)
   - Run: `git init`

2. **Create .gitignore**:
   Create a file named `.gitignore` with:
   ```
   venv/
   __pycache__/
   *.pyc
   *.pkl
   .env
   .DS_Store
   *.log
   ```

3. **Stage and Commit**:
   - Click on Source Control icon (left sidebar)
   - Click "+" next to files to stage them
   - Enter commit message: "Initial commit: Customer Churn Prediction System"
   - Click the checkmark to commit

4. **Connect to GitHub**:
   ```bash
   git remote add origin https://github.com/HayaShahid/customer-churn.git
   git branch -M main
   git push -u origin main
   ```

### Method 2: Using Terminal

```bash
# Initialize repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Customer Churn Prediction System"

# Connect to GitHub
git remote add origin https://github.com/HayaShahid/customer-churn.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Updating Your Repository Later

```bash
# Stage changes
git add .

# Commit changes
git commit -m "Your commit message here"

# Push to GitHub
git push
```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License.

##  Author

**Haya Shahid**
- GitHub: [@HayaShahid](https://github.com/HayaShahid)

##  Acknowledgments

- Dataset: Telecom Churn Dataset from Kaggle (https://www.kaggle.com/code/mnassrib/customer-churn-prediction-telecom-churn-dataset)
- Streamlit for the amazing dashboard framework
- SHAP library for model interpretability

##  Contact

For questions or feedback, please open an issue on GitHub.

---

** If you find this project helpful, please consider giving it a star!**