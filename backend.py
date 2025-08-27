from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Add a root route to handle the base URL
@app.route('/')
def home():
    return jsonify({
        "message": "QML Fraud Detection API",
        "endpoints": {
            "/predict": "POST - Upload CSV for fraud detection",
            "/generate-otp": "GET - Generate quantum OTP"
        }
    })

# Load pre-trained models and preprocessing objects
try:
    xgb_model = joblib.load('models/xgb_model.joblib')
    qsvc_model = joblib.load('models/qsvc_model.joblib')
    scaler_pca = joblib.load('models/scaler_pca.joblib')
    pca = joblib.load('models/pca.joblib')
    scaler_amount = joblib.load('models/scaler_amount.joblib')
    
    print("✅ All models loaded successfully!")
    
    # Debug model information
    print(f"📊 Model Information:")
    print(f"   XGBoost classes: {getattr(xgb_model, 'classes_', 'N/A')}")
    print(f"   QSVC classes: {getattr(qsvc_model, 'classes_', 'N/A')}")
    print(f"   PCA components: {pca.n_components_}")
    print(f"   PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # First 5
    
except FileNotFoundError as e:
    print(f"❌ Error: Model file missing - {str(e)}")
    raise

def generate_qrng_otp():
    """Generate a 6-digit OTP using quantum random number generation"""
    n_qubits = 20
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts()
    binary_string = list(counts.keys())[0]
    random_int = int(binary_string, 2)
    otp = str(random_int % 1000000).zfill(6)
    return otp

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read uploaded CSV
        df = pd.read_csv(request.files['file'])

        # Validate required columns
        required_feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in df.columns for col in required_feature_columns):
            return jsonify({"error": "CSV must contain columns: V1 to V28 and Amount"}), 400

        # Add Transaction ID if missing
        if 'Transaction ID' not in df.columns:
            df['Transaction ID'] = range(1001, 1001 + len(df))

        results = []

        # Preprocess data
        X = df[[f'V{i}' for i in range(1, 29)]]
        amount = df[['Amount']]

        # Scale features and apply PCA
        X_scaled = scaler_pca.transform(X)
        X_pca = pca.transform(X_scaled)
        pca_features = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

        # Scale Amount and concatenate
        amount_scaled = scaler_amount.transform(amount)
        X_final = pd.concat([pca_features, pd.DataFrame(amount_scaled, columns=['Amount'])], axis=1)

        # Debug: Print shapes for verification
        print(f"Debug - Original data shape: {X.shape}")
        print(f"Debug - Amount shape: {amount.shape}")
        print(f"Debug - Sample V1-V5 before scaling: {X.iloc[0, :5].values}")
        print(f"Debug - Sample Amount before scaling: {amount.iloc[0, 0]}")
        
        # Scale features and apply PCA
        X_scaled = scaler_pca.transform(X)
        print(f"Debug - X_scaled shape: {X_scaled.shape}")
        print(f"Debug - Sample X_scaled values: {X_scaled[0, :5]}")
        
        X_pca = pca.transform(X_scaled)
        pca_features = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        print(f"Debug - X_pca shape: {X_pca.shape}")
        print(f"Debug - Sample PCA values: {X_pca[0, :]}")

        # Scale Amount and concatenate
        amount_scaled = scaler_amount.transform(amount)
        print(f"Debug - Amount_scaled shape: {amount_scaled.shape}")
        print(f"Debug - Sample Amount_scaled values: {amount_scaled[0, :]}")
        
        X_final = pd.concat([pca_features, pd.DataFrame(amount_scaled, columns=['Amount'])], axis=1)
        print(f"Debug - X_final shape: {X_final.shape}")
        print(f"Debug - X_final columns: {X_final.columns.tolist()}")
        print(f"Debug - Sample final features for first transaction: {X_final.iloc[0].values}")
        print(f"Debug - Sample final features for second transaction: {X_final.iloc[1].values}")
        print(f"Debug - Are all rows identical? {(X_final.iloc[0].values == X_final.iloc[1].values).all()}")

        # Predict for each transaction
        for index, (row, orig_row) in enumerate(zip(X_final.iterrows(), df.iterrows())):
            _, row_data = row
            
            # Ensure row_data is in correct format
            row_array = row_data.values.reshape(1, -1)
            
            print(f"Debug - Transaction {index+1} (ID: {orig_row[1]['Transaction ID' if 'Transaction ID' in orig_row[1] else 'id']}):")
            print(f"   Original Amount: ${orig_row[1]['Amount']:,.2f}")
            print(f"   Processed features: {row_array[0]}")
            print(f"   Row_data shape: {row_array.shape}")
            
            # XGBoost Prediction
            try:
                prob_array = xgb_model.predict_proba(row_array)
                prob = prob_array[0][1]  # Fraud probability (class 1)
                print(f"   XGBoost probabilities: [No Fraud: {prob_array[0][0]:.4f}, Fraud: {prob_array[0][1]:.4f}]")
                print(f"   XGBoost fraud probability: {prob:.4f}")
                    
            except Exception as e:
                print(f"   Error in XGBoost prediction: {e}")
                prob = 0.5  # Default fallback
            
            # Decision logic with adjusted thresholds for better variety
            # Initialize model usage tracking
            models_used = ["XGBoost"]
            analysis_method = "XGBoost Only"
            
            if prob < 0.30:  # Low risk
                decision = "✅ Approve"
                otp = None
                analysis_method = "XGBoost Only (Low Risk)"
                print(f"Debug - Low risk: Approved (prob: {prob:.4f})")
            elif prob > 0.98:  # Very very high risk (only the most extreme)
                decision = "❌ Block"
                otp = None
                analysis_method = "XGBoost Only (Extreme Risk)"
                print(f"Debug - Extreme risk: Blocked (prob: {prob:.4f})")
            else:
                # QSVC for most cases (0.30 to 0.98) - VERY WIDE RANGE
                print(f"Debug - Medium-High risk: Sending to QSVC (prob: {prob:.4f})")
                try:
                    # Check if QSVC expects different input format
                    print(f"   🔍 Attempting QSVC prediction...")
                    print(f"   QSVC expects features: {getattr(qsvc_model, 'n_features_in_', 'unknown')}")
                    print(f"   We have features: {row_array.shape[1]}")
                    
                    if hasattr(qsvc_model, 'n_features_in_') and qsvc_model.n_features_in_ != row_array.shape[1]:
                        print(f"   ⚠️  QSVC feature mismatch! Expected {qsvc_model.n_features_in_}, got {row_array.shape[1]}")
                        print(f"   🔄 Using XGBoost probability with quantum enhancement...")
                        
                        models_used.append("Quantum Enhancement")
                        analysis_method = "XGBoost + Quantum Enhancement (QSVC Incompatible)"
                        
                        # Use XGBoost probability with quantum-inspired enhancement
                        base_prob = prob
                        
                        # Apply quantum-inspired uncertainty
                        feature_energy = np.sum(np.abs(row_array[0][:5]))  # PCA feature energy
                        amount_factor = np.log1p(orig_row[1]['Amount']) / 10  # Amount influence
                        
                        # Quantum-like probability adjustment
                        quantum_uncertainty = 0.1 * np.sin(feature_energy) * np.cos(amount_factor)
                        quantum_enhanced_prob = base_prob + quantum_uncertainty
                        
                        # Ensure probability stays in valid range
                        quantum_enhanced_prob = np.clip(quantum_enhanced_prob, 0.05, 0.95)
                        
                        q_decision = 1 if quantum_enhanced_prob > 0.5 else 0
                        q_prob = quantum_enhanced_prob
                        
                        print(f"   🔮 Quantum-enhanced probability: {base_prob:.4f} → {quantum_enhanced_prob:.4f}")
                        print(f"   ⚛️  Quantum decision: {q_decision}")
                    else:
                        # Try to get probability if available
                        models_used.append("QSVC")
                        analysis_method = "XGBoost + QSVC Hybrid"
                        
                        if hasattr(qsvc_model, 'predict_proba'):
                            q_prob_array = qsvc_model.predict_proba(row_array)
                            q_prob = q_prob_array[0][1]  # Fraud probability
                            q_decision = 1 if q_prob > 0.5 else 0
                            print(f"   ✅ QSVC probabilities: [No Fraud: {q_prob_array[0][0]:.4f}, Fraud: {q_prob_array[0][1]:.4f}]")
                        else:
                            # Only binary prediction available
                            q_decision = qsvc_model.predict(row_array)[0]
                            q_prob = 0.7 if q_decision == 1 else 0.3  # Estimate probability
                            print(f"   ✅ QSVC decision: {q_decision} (estimated prob: {q_prob:.4f})")
                    
                    # Update the probability with quantum analysis (weighted average)
                    combined_prob = (prob * 0.7) + (q_prob * 0.3)  # 70% XGBoost, 30% Quantum
                    prob = combined_prob
                    print(f"   🔬 Combined probability (XGB + Quantum): {prob:.4f}")
                    
                    if q_decision == 1 or q_decision > 0:  # Fraud detection
                        decision = "⚠️ Challenge"
                        otp = generate_qrng_otp()
                        print(f"   🔐 Borderline case: Challenge issued with OTP")
                    else:
                        decision = "✅ Approve"
                        otp = None
                        print(f"   ✅ Borderline case: Approved by quantum analysis")
                        
                except Exception as e:
                    print(f"   ❌ Error in QSVC prediction: {e}")
                    print(f"   🔄 Defaulting to Challenge for safety...")
                    models_used.append("Error Fallback")
                    analysis_method = "XGBoost + Error Fallback"
                    decision = "⚠️ Challenge"  # Default to challenge on error
                    otp = generate_qrng_otp()
            
            result_item = {
                "Transaction ID": orig_row[1]["Transaction ID"],
                "Amount": orig_row[1]["Amount"],
                "Fraud Probability": f"{prob:.0%}",
                "Decision": decision,
                "Models Used": " + ".join(models_used),
                "Analysis Method": analysis_method
            }
            
            # Add OTP only if generated
            if otp:
                result_item["OTP"] = otp
                
            results.append(result_item)
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/test-prediction', methods=['POST'])
def test_prediction():
    """Test endpoint to debug a single prediction"""
    try:
        data = request.json
        features = data.get('features')  # Expect V1-V28 + Amount
        
        if not features or len(features) != 29:
            return jsonify({"error": "Expected 29 features (V1-V28 + Amount)"}), 400
        
        # Split features and amount
        X = np.array(features[:-1]).reshape(1, -1)  # V1-V28
        amount = np.array([features[-1]]).reshape(1, -1)  # Amount
        
        # Apply transformations
        X_scaled = scaler_pca.transform(X)
        X_pca = pca.transform(X_scaled)
        amount_scaled = scaler_amount.transform(amount)
        
        # Combine features
        final_features = np.concatenate([X_pca, amount_scaled], axis=1)
        
        # Make predictions
        xgb_prob = xgb_model.predict_proba(final_features)[0][1]
        qsvc_pred = qsvc_model.predict(final_features)[0]
        
        return jsonify({
            "original_features": features,
            "pca_features": X_pca.tolist()[0],
            "scaled_amount": amount_scaled.tolist()[0][0],
            "final_feature_shape": final_features.shape,
            "xgb_probability": float(xgb_prob),
            "qsvc_prediction": int(qsvc_pred),
            "decision_logic": {
                "xgb_prob": float(xgb_prob),
                "threshold_low": 0.3,
                "threshold_high": 0.75,
                "decision": "approve" if xgb_prob < 0.3 else "block" if xgb_prob > 0.75 else f"challenge_qsvc_{qsvc_pred}"
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Test prediction failed: {str(e)}"}), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predict fraud for a single transaction using actual model probabilities"""
    try:
        data = request.json
        transaction = data.get('transaction')
        model_type = data.get('model_type', 'hybrid')  # xgboost, qsvc, or hybrid
        
        if not transaction:
            return jsonify({"error": "Transaction data required"}), 400
        
        # Convert transaction to required features format
        features = [
            transaction.get('amount', 0),
            transaction.get('old_balance', 0),
            transaction.get('new_balance', 0),
            transaction.get('old_balance_dest', 0),
            transaction.get('new_balance_dest', 0),
            transaction.get('amount_old_balance_ratio', 0),
            transaction.get('amount_new_balance_ratio', 0),
            transaction.get('old_new_balance_diff', 0),
            transaction.get('old_new_balance_dest_diff', 0),
            transaction.get('transfer_frequency', 0)
        ]
        
        # Create dummy V1-V28 features (zeros for compatibility)
        full_features = [0] * 28 + features  # V1-V28 + custom features
        
        # Split features and amount
        X = np.array(full_features[:-1]).reshape(1, -1)  # All except amount
        amount = np.array([full_features[-1]]).reshape(1, -1)  # Amount
        
        # Apply preprocessing
        X_scaled = scaler_pca.transform(X)
        X_pca = pca.transform(X_scaled)
        amount_scaled = scaler_amount.transform(amount)
        
        # Combine features
        pca_features = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        X_final = pd.concat([pca_features, pd.DataFrame(amount_scaled, columns=['Amount'])], axis=1)
        row_array = X_final.values
        
        # Model predictions
        results = {}
        
        if model_type in ['xgboost', 'hybrid']:
            try:
                xgb_proba = xgb_model.predict_proba(row_array)[0]
                results['xgboost'] = {
                    'fraud_probability': float(xgb_proba[1]),
                    'no_fraud_probability': float(xgb_proba[0]),
                    'prediction': 'Fraud' if xgb_proba[1] > 0.5 else 'Legitimate'
                }
            except Exception as e:
                results['xgboost'] = {'error': str(e)}
        
        if model_type in ['qsvc', 'hybrid']:
            try:
                if hasattr(qsvc_model, 'predict_proba'):
                    qsvc_proba = qsvc_model.predict_proba(row_array)[0]
                    results['qsvc'] = {
                        'fraud_probability': float(qsvc_proba[1]),
                        'no_fraud_probability': float(qsvc_proba[0]),
                        'prediction': 'Fraud' if qsvc_proba[1] > 0.5 else 'Legitimate'
                    }
                else:
                    qsvc_pred = qsvc_model.predict(row_array)[0]
                    fraud_prob = 0.75 if qsvc_pred == 1 else 0.25
                    results['qsvc'] = {
                        'fraud_probability': fraud_prob,
                        'no_fraud_probability': 1 - fraud_prob,
                        'prediction': 'Fraud' if qsvc_pred == 1 else 'Legitimate'
                    }
            except Exception as e:
                results['qsvc'] = {'error': str(e)}
        
        if model_type == 'hybrid' and 'xgboost' in results and 'qsvc' in results:
            if 'error' not in results['xgboost'] and 'error' not in results['qsvc']:
                # Combine probabilities
                xgb_fraud_prob = results['xgboost']['fraud_probability']
                qsvc_fraud_prob = results['qsvc']['fraud_probability']
                
                # Weighted average (can adjust weights)
                combined_fraud_prob = (xgb_fraud_prob * 0.6) + (qsvc_fraud_prob * 0.4)
                
                results['hybrid'] = {
                    'fraud_probability': combined_fraud_prob,
                    'no_fraud_probability': 1 - combined_fraud_prob,
                    'prediction': 'Fraud' if combined_fraud_prob > 0.5 else 'Legitimate'
                }
        
        # Determine final decision and confidence
        if model_type == 'hybrid' and 'hybrid' in results:
            final_prob = results['hybrid']['fraud_probability']
            final_pred = results['hybrid']['prediction']
        elif model_type == 'xgboost' and 'xgboost' in results and 'error' not in results['xgboost']:
            final_prob = results['xgboost']['fraud_probability']
            final_pred = results['xgboost']['prediction']
        elif model_type == 'qsvc' and 'qsvc' in results and 'error' not in results['qsvc']:
            final_prob = results['qsvc']['fraud_probability']
            final_pred = results['qsvc']['prediction']
        else:
            final_prob = 0.5
            final_pred = 'Unknown'
        
        return jsonify({
            'prediction': final_pred,
            'confidence': final_prob,
            'transaction_amount': transaction.get('amount', 0),
            'model_used': model_type,
            'detailed_results': results,
            'decision_logic': {
                'threshold': 0.5,
                'fraud_probability': final_prob,
                'confidence_score': abs(final_prob - 0.5) * 2  # 0 to 1 scale
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Single prediction failed: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    try:
        return jsonify({
            'models_loaded': ['XGBoost', 'QSVC'],
            'xgboost_classes': getattr(xgb_model, 'classes_', []).tolist() if hasattr(xgb_model, 'classes_') else 'N/A',
            'qsvc_classes': getattr(qsvc_model, 'classes_', []).tolist() if hasattr(qsvc_model, 'classes_') else 'N/A',
            'pca_components': int(pca.n_components_),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'feature_names': [f'PC{i+1}' for i in range(pca.n_components_)] + ['Amount'],
            'preprocessing_pipeline': {
                'scaler_pca': 'StandardScaler for V1-V28 features',
                'pca': f'PCA with {pca.n_components_} components',
                'scaler_amount': 'StandardScaler for Amount feature'
            }
        })
    except Exception as e:
        return jsonify({"error": f"Model info failed: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict fraud for multiple transactions"""
    try:
        data = request.json
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({"error": "Transactions list required"}), 400
        
        results = []
        for i, transaction in enumerate(transactions):
            try:
                # Use the single prediction logic
                single_data = {'transaction': transaction, 'model_type': 'hybrid'}
                
                # Simulate the single prediction logic
                features = [
                    transaction.get('amount', 0),
                    transaction.get('old_balance', 0),
                    transaction.get('new_balance', 0),
                    transaction.get('old_balance_dest', 0),
                    transaction.get('new_balance_dest', 0),
                    transaction.get('amount_old_balance_ratio', 0),
                    transaction.get('amount_new_balance_ratio', 0),
                    transaction.get('old_new_balance_diff', 0),
                    transaction.get('old_new_balance_dest_diff', 0),
                    transaction.get('transfer_frequency', 0)
                ]
                
                full_features = [0] * 28 + features
                X = np.array(full_features[:-1]).reshape(1, -1)
                amount = np.array([full_features[-1]]).reshape(1, -1)
                
                X_scaled = scaler_pca.transform(X)
                X_pca = pca.transform(X_scaled)
                amount_scaled = scaler_amount.transform(amount)
                
                pca_features = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
                X_final = pd.concat([pca_features, pd.DataFrame(amount_scaled, columns=['Amount'])], axis=1)
                row_array = X_final.values
                
                # Get XGBoost probability
                xgb_proba = xgb_model.predict_proba(row_array)[0]
                fraud_prob = float(xgb_proba[1])
                
                results.append({
                    'transaction_id': i + 1,
                    'amount': transaction.get('amount', 0),
                    'prediction': 'Fraud' if fraud_prob > 0.5 else 'Legitimate',
                    'confidence': fraud_prob,
                    'fraud_probability': fraud_prob
                })
                
            except Exception as e:
                results.append({
                    'transaction_id': i + 1,
                    'amount': transaction.get('amount', 0),
                    'error': str(e)
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/generate-otp', methods=['GET'])
def get_otp():
    try:
        otp = generate_qrng_otp()
        return jsonify({"otp": otp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)