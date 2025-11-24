import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import shap
import pickle
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue and white theme
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #F0F8FF;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .metric-card {
        background-color: #EFF6FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #60A5FA;'>Predict customer churn and take proactive retention actions</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/business-analytics.png", width=80)
    st.markdown("## 🎯 Navigation")
    page = st.radio("Select Page:", 
                    ["Dashboard", "Model Training", "Prediction", "Model Insights", "Recommendations"])
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This system uses machine learning to predict customer churn and provide actionable insights for retention strategies.")

# Load or initialize data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Data/telecom_churn.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please ensure 'telecom_churn.csv' is in the 'Data' folder.")
        return None

# Preprocess data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Convert categorical to numeric
    df_processed['International plan'] = df_processed['International plan'].map({'Yes': 1, 'No': 0})
    df_processed['Voice mail plan'] = df_processed['Voice mail plan'].map({'Yes': 1, 'No': 0})
    df_processed['Churn'] = df_processed['Churn'].astype(int)
    
    # Drop State and Area code (high cardinality)
    df_processed = df_processed.drop(['State', 'Area code'], axis=1)
    
    return df_processed

# Train model
def train_model(df, model_type='RandomForest'):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        model = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, metrics

# Dashboard Page
if page == "Dashboard":
    df = load_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        churned = df['Churn'].sum()
        churn_rate = (churned / total_customers) * 100
        retained = total_customers - churned
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("👥 Total Customers", f"{total_customers:,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("❌ Churned", f"{churned:,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("✅ Retained", f"{retained:,}", delta=f"{100-churn_rate:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("📊 Churn Rate", f"{churn_rate:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Churn Distribution")
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Retained', 'Churned'],
                values=[churn_counts[0], churn_counts[1]],
                marker=dict(colors=["#5394FD", "#F64646"]),
                hole=0.4
            )])
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Customer Service Calls vs Churn")
            fig = px.histogram(df, x='Customer service calls', color='Churn',
                             color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                             barmode='group')
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### International Plan Impact")
            intl_churn = df.groupby('International plan')['Churn'].mean() * 100
            fig = px.bar(x=['No Plan', 'Has Plan'], y=intl_churn.values,
                        labels={'x': 'International Plan', 'y': 'Churn Rate (%)'},
                        color=intl_churn.values,
                        color_continuous_scale=['#3B82F6', '#EF4444'])
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("###  Total Day Charge Distribution")
            fig = px.box(df, x='Churn', y='Total day charge',
                        color='Churn',
                        color_discrete_map={0: '#3B82F6', 1: '#EF4444'})
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "Model Training":
    st.markdown("##  Train Your Churn Prediction Model")
    
    df = load_data()
    
    if df is not None:
        df_processed = preprocess_data(df)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Model Configuration")
            model_type = st.selectbox("Select Model:", ["RandomForest", "XGBoost"])
            
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model... Please wait."):
                    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, metrics = train_model(df_processed, model_type)
                    
                    # Save model
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    
                    st.session_state['model'] = model
                    st.session_state['metrics'] = metrics
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    st.session_state['y_pred_proba'] = y_pred_proba
                    st.session_state['feature_names'] = X_test.columns.tolist()
                    
                    st.success("✅ Model trained successfully!")
        
        with col2:
            if 'metrics' in st.session_state:
                st.markdown("### Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{st.session_state['metrics']['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{st.session_state['metrics']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{st.session_state['metrics']['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{st.session_state['metrics']['f1']:.3f}")
                
                # Confusion Matrix
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted No Churn', 'Predicted Churn'],
                    y=['Actual No Churn', 'Actual Churn'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16}
                ))
                fig.update_layout(title="Confusion Matrix", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(st.session_state['y_test'], st.session_state['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})',
                                        line=dict(color='#2563EB', width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                        line=dict(color='gray', width=2, dash='dash')))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate', height=400)
                st.plotly_chart(fig, use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.markdown("##  Predict Customer Churn")
    
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        st.markdown("### Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            account_length = st.number_input("Account Length (days)", min_value=1, max_value=300, value=100)
            intl_plan = st.selectbox("International Plan", ["No", "Yes"])
            voice_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
            vmail_messages = st.number_input("Voice Mail Messages", min_value=0, max_value=60, value=0)
            
        with col2:
            day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
            day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
            day_charge = st.number_input("Total Day Charge ($)", min_value=0.0, max_value=70.0, value=30.0)
            eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
            
        with col3:
            eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
            eve_charge = st.number_input("Total Evening Charge ($)", min_value=0.0, max_value=40.0, value=17.0)
            night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
            night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            night_charge = st.number_input("Total Night Charge ($)", min_value=0.0, max_value=20.0, value=9.0)
        with col2:
            intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=20.0, value=10.0)
            intl_calls_input = st.number_input("Total International Calls", min_value=0, max_value=20, value=4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            intl_charge = st.number_input("Total International Charge ($)", min_value=0.0, max_value=6.0, value=2.7)
        with col2:
            service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
        
        if st.button(" Predict Churn", use_container_width=True):
            input_data = pd.DataFrame({
                'Account length': [account_length],
                'International plan': [1 if intl_plan == 'Yes' else 0],
                'Voice mail plan': [1 if voice_plan == 'Yes' else 0],
                'Number vmail messages': [vmail_messages],
                'Total day minutes': [day_minutes],
                'Total day calls': [day_calls],
                'Total day charge': [day_charge],
                'Total eve minutes': [eve_minutes],
                'Total eve calls': [eve_calls],
                'Total eve charge': [eve_charge],
                'Total night minutes': [night_minutes],
                'Total night calls': [night_calls],
                'Total night charge': [night_charge],
                'Total intl minutes': [intl_minutes],
                'Total intl calls': [intl_calls_input],
                'Total intl charge': [intl_charge],
                'Customer service calls': [service_calls]
            })
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK: Customer Likely to Churn")
                    st.markdown(f"**Churn Probability:** {probability[1]*100:.1f}%")
                else:
                    st.success("### ✅ LOW RISK: Customer Likely to Stay")
                    st.markdown(f"**Retention Probability:** {probability[0]*100:.1f}%")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1]*100,
                    title={'text': "Churn Risk Score"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "#EF4444" if prediction == 1 else "#3B82F6"},
                           'steps': [
                               {'range': [0, 30], 'color': "#DBEAFE"},
                               {'range': [30, 70], 'color': "#93C5FD"},
                               {'range': [70, 100], 'color': "#FCA5A5"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75,
                                       'value': 50}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please train a model first in the 'Model Training' page!")

# Model Insights Page
elif page == "Model Insights":
    st.markdown("##  Model Interpretability with SHAP")
    
    if os.path.exists('model.pkl') and 'X_test' in st.session_state:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with st.spinner("Calculating SHAP values... This may take a moment."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(st.session_state['X_test'])
            
            st.markdown("### Feature Importance")
            
            feature_importance = pd.DataFrame({
                'feature': st.session_state['X_test'].columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                        orientation='h', color='importance',
                        color_continuous_scale='Blues')
            fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### SHAP Summary Plot")
            st.info("This plot shows which features have the most impact on predictions. Red indicates high feature values, blue indicates low values.")
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, use the positive class (index 1)
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values_to_plot, st.session_state['X_test'], plot_type="dot", show=False)
            st.pyplot(fig)
            plt.close()
            
    else:
        st.warning("⚠️ Please train a model first in the 'Model Training' page!")

# Recommendations Page
else:
    st.markdown("## Retention Strategies & Recommendations")
    
    st.markdown("""
    <div style='background-color: #EFF6FF; padding: 20px; border-radius: 10px; border-left: 5px solid #2563EB;'>
    <h3 style='color: #1E3A8A;'>🎯 Key Insights for Reducing Churn</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Customer Service Focus")
        st.markdown("""
        - **High-Risk Indicator:** Customers with 4+ service calls
        - **Action:** Implement proactive support for frequent callers
        - **Solution:** Assign dedicated account managers
        - **Expected Impact:** 15-20% churn reduction
        """)
        
        st.markdown("### International Plan Strategy")
        st.markdown("""
        - **Observation:** Higher churn among international plan users
        - **Action:** Review pricing and competitive offerings
        - **Solution:** Introduce flexible international packages
        - **Expected Impact:** 10-15% churn reduction
        """)
    
    with col2:
        st.markdown("### Pricing Optimization")
        st.markdown("""
        - **Target:** Customers with high day charges
        - **Action:** Offer loyalty discounts and custom plans
        - **Solution:** Usage-based pricing tiers
        - **Expected Impact:** 12-18% churn reduction
        """)
        
        st.markdown("### Engagement Programs")
        st.markdown("""
        - **Strategy:** Early warning system for at-risk customers
        - **Action:** Automated retention campaigns
        - **Solution:** Personalized offers based on usage patterns
        - **Expected Impact:** 20-25% churn reduction
        """)
    
    st.markdown("---")
    
    st.markdown("### Immediate Action Items")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #DBEAFE; padding: 15px; border-radius: 10px; text-align: center;'>
        <h4 style='color: #1E3A8A;'>Week 1</h4>
        <p>Identify high-risk customers and contact them</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #DBEAFE; padding: 15px; border-radius: 10px; text-align: center;'>
        <h4 style='color: #1E3A8A;'>Week 2-3</h4>
        <p>Launch targeted retention campaigns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #DBEAFE; padding: 15px; border-radius: 10px; text-align: center;'>
        <h4 style='color: #1E3A8A;'>Month 1+</h4>
        <p>Monitor results and optimize strategies</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #60A5FA;'>Built with using Streamlit | Customer Churn Prediction System</p>", unsafe_allow_html=True)