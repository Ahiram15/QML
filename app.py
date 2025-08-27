import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🔐 QML Fraud Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background and quantum theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a2e 0%, #16213e 50%, #1a1a3a 100%);
        color: #ffffff;
    }
    
    /* Custom header styling */
    .quantum-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem auto;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
        position: relative;
        z-index: 10;
        width: 100%;
        max-width: 1200px;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3); }
        to { box-shadow: 0 20px 60px rgba(139, 92, 246, 0.5); }
    }
    
    .quantum-title {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        display: block !important;
        visibility: visible !important;
        position: relative;
        z-index: 11;
    }
    
    .quantum-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        margin-top: 1rem;
        opacity: 0.9;
        color: #ffffff !important;
        display: block !important;
        visibility: visible !important;
        position: relative;
        z-index: 11;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px dashed #6366f1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Force metric values to be white and visible */
    .stMetric {
        color: #ffffff !important;
    }
    
    .stMetric > div {
        color: #ffffff !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    
    .stMetric [data-testid="metric-container"] > div > div {
        color: #ffffff !important;
    }
    
    .stMetric [data-testid="metric-container"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    /* Force all metric text to be white */
    div[data-testid="metric-container"] * {
        color: #ffffff !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Quantum OTP display */
    .quantum-otp {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.8);
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border-color: #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Ensure tab text is white */
    .stTabs [data-baseweb="tab"] > div {
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Ensure quantum header is always visible */
    div[data-testid="stMarkdownContainer"] .quantum-header {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 999 !important;
    }
    
    /* Force header content to be visible */
    .quantum-header * {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Override any Streamlit hiding */
    .stMarkdown .quantum-header {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="quantum-header">
    <h1 class="quantum-title">⚛️ QML FRAUD DETECTION</h1>
    <p class="quantum-subtitle">Quantum-Enhanced Security • Real-time Analysis • AI-Powered Protection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with system status
with st.sidebar:
    st.markdown("### 🔧 System Status")
    
    # Check backend connectivity
    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=3)
        if response.status_code == 200:
            st.success("🟢 Backend Online")
            st.markdown("**API Endpoints:**")
            st.markdown("• Fraud Detection: Active")
            st.markdown("• Quantum OTP: Ready")
        else:
            st.error("🔴 Backend Error")
    except:
        st.error("🔴 Backend Offline")
        st.warning("Start the Flask server:\n```python backend.py```")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if 'transaction_data' in st.session_state and st.session_state.transaction_data is not None:
        df = st.session_state.transaction_data
        total = len(df)
        approved = len(df[df['Decision'] == '✅ Approve'])
        blocked = len(df[df['Decision'] == '❌ Block'])
        challenged = len(df[df['Decision'] == '⚠️ Challenge'])
        
        st.metric("Total Transactions", total)
        st.metric("Approved", approved, delta=f"{(approved/total*100):.1f}%")
        st.metric("Blocked", blocked, delta=f"{(blocked/total*100):.1f}%")
        st.metric("Challenged", challenged, delta=f"{(challenged/total*100):.1f}%")
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    if st.button("🧹 Clear Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Initialize session state
if 'challenge_transaction' not in st.session_state:
    st.session_state.challenge_transaction = None
if 'otp' not in st.session_state:
    st.session_state.otp = None
if 'transaction_data' not in st.session_state:
    st.session_state.transaction_data = None

# Dashboard Screen
def display_dashboard():
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Analytics Dashboard", "🔍 Transaction Details"])
    
    with tab1:
        st.markdown("### 📤 Upload Transaction Data")
        
        # File upload with enhanced styling
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your transaction CSV file",
                type=["csv"],
                help="Upload a CSV file containing transaction data with V1-V28 features and Amount column"
            )
        
        with col2:
            st.markdown("**Required Format:**")
            st.markdown("• V1 to V28 (PCA features)")
            st.markdown("• Amount column")
            st.markdown("• Optional: Transaction ID")
        
        if uploaded_file is not None:
            # Show upload progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📤 Uploading file...")
                progress_bar.progress(25)
                
                files = {'file': uploaded_file.getvalue()}
                
                status_text.text("🔍 Analyzing transactions...")
                progress_bar.progress(50)
                
                response = requests.post("http://127.0.0.1:5000/predict", files=files, timeout=30)
                progress_bar.progress(75)
                
                if response.status_code == 200:
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    time.sleep(1)
                    
                    data = response.json()
                    st.session_state.transaction_data = pd.DataFrame(data)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Successfully analyzed {len(data)} transactions!")
                    
                    # Display model usage summary
                    display_model_usage_summary(pd.DataFrame(data))
                    
                else:
                    st.error(f"❌ Prediction failed: {response.json().get('error', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend at http://127.0.0.1:5000. Ensure the Flask server is running.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The server might be overloaded.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        if st.session_state.transaction_data is not None:
            display_analytics_dashboard()
        else:
            st.info("📊 Upload transaction data to view analytics")
    
    with tab3:
        if st.session_state.transaction_data is not None:
            display_transaction_details()
        else:
            st.info("🔍 Upload transaction data to view details")

def display_model_usage_summary(df):
    """Display summary of which models were used for analysis"""
    st.markdown("### 🤖 Model Usage Analysis")
    
    # Count model usage
    if 'Models Used' in df.columns and 'Analysis Method' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 Models Used")
            model_counts = {}
            for models_str in df['Models Used']:
                models = models_str.split(' + ')
                for model in models:
                    model_counts[model] = model_counts.get(model, 0) + 1
            
            for model, count in model_counts.items():
                percentage = (count / len(df)) * 100
                if model == 'XGBoost':
                    emoji = "🌳"
                elif model == 'QSVC':
                    emoji = "⚛️"
                elif model == 'Quantum Enhancement':
                    emoji = "🔮"
                else:
                    emoji = "🔧"
                
                st.markdown(f"{emoji} **{model}**: {count} transactions ({percentage:.1f}%)")
        
        with col2:
            st.markdown("#### 📊 Analysis Methods")
            method_counts = df['Analysis Method'].value_counts()
            
            for method, count in method_counts.items():
                percentage = (count / len(df)) * 100
                
                if 'XGBoost Only' in method:
                    emoji = "🌳"
                    color = "blue"
                elif 'QSVC Hybrid' in method:
                    emoji = "⚛️"
                    color = "green"
                elif 'Quantum Enhancement' in method:
                    emoji = "🔮"
                    color = "purple"
                else:
                    emoji = "🔧"
                    color = "gray"
                
                st.markdown(f"{emoji} **{method}**: {count} ({percentage:.1f}%)")
        
        # Model flow visualization
        st.markdown("#### 🔄 Decision Flow")
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.8); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <p><strong>🌳 XGBoost</strong> → Initial fraud probability assessment</p>
            <p>├─ <strong>Low Risk (< 30%)</strong> → ✅ Direct Approval</p>
            <p>├─ <strong>Extreme Risk (> 98%)</strong> → ❌ Direct Block</p>
            <p>└─ <strong>Medium Risk (30-98%)</strong> → Quantum Analysis</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;├─ <strong>⚛️ QSVC Available</strong> → Hybrid Analysis</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;└─ <strong>🔮 QSVC Incompatible</strong> → Quantum Enhancement</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model usage information not available in the response")

def display_analytics_dashboard():
    """Display analytics dashboard with charts and metrics"""
    df = st.session_state.transaction_data
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    approved = len(df[df['Decision'] == '✅ Approve'])
    blocked = len(df[df['Decision'] == '❌ Block'])
    challenged = len(df[df['Decision'] == '⚠️ Challenge'])
    
    with col1:
        st.metric("🔢 Total Transactions", total)
    with col2:
        st.metric("✅ Approved", approved, delta=f"{(approved/total*100):.1f}%")
    with col3:
        st.metric("❌ Blocked", blocked, delta=f"{(blocked/total*100):.1f}%")
    with col4:
        st.metric("⚠️ Challenged", challenged, delta=f"{(challenged/total*100):.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Decision distribution pie chart
        decision_counts = df['Decision'].value_counts()
        fig_pie = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="🎯 Decision Distribution",
            color_discrete_map={
                '✅ Approve': '#10b981',
                '❌ Block': '#ef4444',
                '⚠️ Challenge': '#f59e0b'
            }
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Fraud probability distribution
        df['Probability_Numeric'] = df['Fraud Probability'].str.replace('%', '').astype(float)
        fig_hist = px.histogram(
            df, x='Probability_Numeric',
            title="📈 Fraud Probability Distribution",
            color_discrete_sequence=['#6366f1']
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Fraud Probability (%)",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def display_transaction_details():
    """Display detailed transaction table with enhanced formatting"""
    df = st.session_state.transaction_data
    
    st.markdown("### 📋 Transaction Details")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        decision_filter = st.selectbox(
            "Filter by Decision:",
            ["All", "✅ Approve", "❌ Block", "⚠️ Challenge"]
        )
    
    with col2:
        if 'Analysis Method' in df.columns:
            method_filter = st.selectbox(
                "Filter by Analysis Method:",
                ["All"] + list(df['Analysis Method'].unique())
            )
        else:
            method_filter = "All"
    
    with col3:
        min_amount = st.number_input("Min Amount:", value=0.0, step=100.0)
    
    with col4:
        max_amount = st.number_input("Max Amount:", value=float(df['Amount'].max()), step=100.0)
    
    # Apply filters
    filtered_df = df.copy()
    if decision_filter != "All":
        filtered_df = filtered_df[filtered_df['Decision'] == decision_filter]
    
    if method_filter != "All" and 'Analysis Method' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Analysis Method'] == method_filter]
    
    filtered_df = filtered_df[
        (filtered_df['Amount'] >= min_amount) & 
        (filtered_df['Amount'] <= max_amount)
    ]
    
    # Enhanced dataframe display with styling
    def style_decision(val):
        if val == "✅ Approve":
            return 'background-color: #10b981; color: white; font-weight: bold;'
        elif val == "⚠️ Challenge":
            return 'background-color: #f59e0b; color: white; font-weight: bold;'
        else:
            return 'background-color: #ef4444; color: white; font-weight: bold;'
    
    styled_df = filtered_df.style.applymap(style_decision, subset=['Decision'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Challenge transaction actions
    challenge_transactions = filtered_df[filtered_df['Decision'] == "⚠️ Challenge"]
    
    if not challenge_transactions.empty:
        st.markdown("### ⚠️ Transactions Requiring Verification")
        
        for index, row in challenge_transactions.iterrows():
            with st.expander(f"🔐 Transaction {row['Transaction ID']} - ${row['Amount']:,.2f}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Fraud Probability:** {row['Fraud Probability']}")
                    if 'Analysis Method' in row:
                        st.write(f"**Analysis Method:** {row['Analysis Method']}")
                    if 'Models Used' in row:
                        st.write(f"**Models Used:** {row['Models Used']}")
                
                with col2:
                    if 'OTP' in row and pd.notna(row['OTP']):
                        st.markdown(f"**🔐 Quantum OTP:** `{row['OTP']}`")
                        st.markdown("*Generated using quantum circuits*")
                    else:
                        st.write("🔄 OTP: Not generated")
                    
                    if st.button(f"🔐 Verify Transaction", key=f"verify_{row['Transaction ID']}"):
                        st.session_state.challenge_transaction = row
                        st.session_state.otp = row.get('OTP')
                        if not st.session_state.otp:
                            # Generate OTP if not available
                            try:
                                otp_response = requests.get("http://127.0.0.1:5000/generate-otp", timeout=10)
                                if otp_response.status_code == 200:
                                    st.session_state.otp = otp_response.json()['otp']
                                else:
                                    st.error("Failed to generate OTP")
                            except Exception as e:
                                st.error(f"OTP generation error: {str(e)}")
                        st.rerun()

# Challenge Screen
def display_challenge_screen():
    transaction = st.session_state.challenge_transaction
    
    # Challenge screen header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);">
        <h2 style="margin: 0; color: white; font-family: 'Orbitron', monospace;">
            ⚠️ TRANSACTION VERIFICATION REQUIRED
        </h2>
        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
            Quantum-Enhanced Security Challenge
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Transaction details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Transaction Details")
        st.info(f"**Transaction ID:** {transaction['Transaction ID']}")
        st.info(f"**Amount:** ${transaction['Amount']:,.2f}")
        st.info(f"**Fraud Probability:** {transaction['Fraud Probability']}")
        
        # Risk assessment visual
        prob_value = float(transaction['Fraud Probability'].replace('%', ''))
        if prob_value >= 60:
            risk_level = "🔴 HIGH RISK"
            risk_color = "#ef4444"
        elif prob_value >= 40:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "#f59e0b"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "#10b981"
        
        st.markdown(f"""
        <div style="background: {risk_color}; padding: 1rem; border-radius: 8px; 
                    text-align: center; color: white; font-weight: bold; margin: 1rem 0;">
            {risk_level}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔐 Quantum OTP Verification")
        
        if st.session_state.otp:
            # Display OTP with quantum styling
            st.markdown(f"""
            <div class="quantum-otp">
                <div style="font-size: 1rem; margin-bottom: 0.5rem;">QUANTUM OTP</div>
                <div>{st.session_state.otp}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                    Generated via Quantum Random Number Generation
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ OTP not available")
        
        # OTP input
        st.markdown("---")
        user_otp = st.text_input(
            "🔑 Enter the 6-digit OTP to verify this transaction:",
            max_chars=6,
            placeholder="000000",
            help="Enter the quantum-generated OTP shown above"
        )
        
        # Verification buttons
        col_verify, col_cancel = st.columns(2)
        
        with col_verify:
            if st.button("✅ Verify OTP", use_container_width=True, type="primary"):
                if user_otp == st.session_state.otp:
                    st.success("🎉 OTP Verified! Transaction Approved.")
                    st.balloons()
                    
                    # Update transaction status
                    st.session_state.transaction_data.loc[
                        st.session_state.transaction_data['Transaction ID'] == transaction['Transaction ID'], 
                        'Decision'
                    ] = "✅ Approve"
                    
                    time.sleep(2)
                    st.session_state.challenge_transaction = None
                    st.session_state.otp = None
                    st.rerun()
                    
                else:
                    st.error("❌ Invalid OTP. Transaction Blocked for security.")
                    
                    # Update transaction status
                    st.session_state.transaction_data.loc[
                        st.session_state.transaction_data['Transaction ID'] == transaction['Transaction ID'], 
                        'Decision'
                    ] = "❌ Block"
                    
                    time.sleep(2)
                    st.session_state.challenge_transaction = None
                    st.session_state.otp = None
                    st.rerun()
        
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.warning("Verification cancelled. Returning to dashboard.")
                st.session_state.challenge_transaction = None
                st.session_state.otp = None
                time.sleep(1)
                st.rerun()
    
    # Security information
    st.markdown("---")
    with st.expander("🔍 How Quantum OTP Works"):
        st.markdown("""
        **Quantum Random Number Generation:**
        - Measures quantum states to generate true randomness
        - Creates cryptographically secure 6-digit OTP
        - Cannot be predicted or reproduced
        
        **Security Benefits:**
        - Quantum uncertainty ensures unpredictability
        - Resistant to classical computing attacks
        - Enhanced security for high-risk transactions
        """)

# Screen Navigation
if st.session_state.challenge_transaction is None:
    display_dashboard()
else:
    display_challenge_screen()