import xgboost as xgb
from flask import Flask, request, render_template, flash, redirect, url_for
import pandas as pd
import pickle
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with error handling
def load_models():
    """Load all required models and preprocessors"""
    try:
        with open('model/LoanElligibilityPredictor.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model/columns.pkl', 'rb') as f:
            expected_columns = pickle.load(f)
        
        logger.info("All models loaded successfully")
        return model, scaler, expected_columns
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Load models at startup
try:
    model, scaler, expected_columns = load_models()
    MODELS_LOADED = True
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model, scaler, expected_columns = None, None, None
    MODELS_LOADED = False

def validate_input_data(input_data):
    """Validate and clean input data"""
    required_fields = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in input_data or input_data[field] == '']
    if missing_fields:
        raise ValueError(f"Please fill in all required fields: {', '.join(missing_fields)}")
    
    # Validate and convert numeric fields
    numeric_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    for key in numeric_fields:
        try:
            value = float(input_data[key])
            if value < 0:
                raise ValueError(f"{key.replace('_', ' ')} cannot be negative")
            if key in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'] and value == 0:
                raise ValueError(f"{key.replace('_', ' ')} cannot be zero")
            input_data[key] = value
        except (ValueError, TypeError) as e:
            if "could not convert" in str(e):
                raise ValueError(f"{key.replace('_', ' ')} must be a valid number")
            raise e
    
    # Validate specific field ranges
    if input_data['Credit_History'] not in [0.0, 1.0]:
        raise ValueError("Credit History must be 0 (No) or 1 (Yes)")
    
    if input_data['Loan_Amount_Term'] < 12:
        raise ValueError("Loan Amount Term must be at least 12 months")
    
    return input_data

def make_prediction(input_data):
    """Process input data and make prediction"""
    if not MODELS_LOADED:
        raise RuntimeError("Prediction models are not available")
    
    # Validate input
    validated_data = validate_input_data(input_data)
    
    # Create DataFrame
    df = pd.DataFrame([validated_data])
    
    # Feature engineering (must match training pipeline)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['IncomePerLoan'] = df['TotalIncome'] / (df['LoanAmount'] + 1e-6)
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df)
    
    # Align columns with training data
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    # Scale features
    scaled_input = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    
    result = {
        'eligible': bool(prediction),
        'status': "✅ Eligible" if prediction == 1 else "❌ Ineligible",
        'confidence': float(max(probability)) * 100,
        'probability_eligible': float(probability[1]) * 100 if len(probability) > 1 else 0
    }
    
    return result

@app.route('/')
def home():
    """Home page with the loan application form"""
    return render_template("index.html")

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission and make prediction"""
    try:
        # Extract form data
        input_data = {k: request.form[k] for k in request.form}
        
        # Log prediction attempt
        logger.info(f"Prediction request received at {datetime.now()}")
        
        # Make prediction
        result = make_prediction(input_data)
        
        # Log successful prediction
        logger.info(f"Prediction successful: {result['status']}")
        
        # Prepare additional info for display
        additional_info = {
            'applicant_income': f"₹{input_data['ApplicantIncome']:,.0f}",
            'loan_amount': f"₹{input_data['LoanAmount']:,.0f}",
            'total_income': f"₹{float(input_data['ApplicantIncome']) + float(input_data['CoapplicantIncome']):,.0f}",
            'emi': f"₹{float(input_data['LoanAmount']) / float(input_data['Loan_Amount_Term']):,.0f}"
        }
        
        return render_template("result.html", 
                             prediction=result['status'],
                             confidence=f"{result['confidence']:.1f}%",
                             probability=f"{result['probability_eligible']:.1f}%",
                             additional_info=additional_info,
                             input_data=input_data)
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        flash(str(e), 'error')
        return redirect(url_for('home'))
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return render_template("result.html", 
                             prediction="❌ Service Unavailable",
                             error="The prediction service is temporarily unavailable. Please try again later.")
    
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        return render_template("result.html", 
                             prediction="❌ Prediction Failed",
                             error="An unexpected error occurred. Please check your input and try again.")

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    status = {
        'status': 'healthy' if MODELS_LOADED else 'unhealthy',
        'models_loaded': MODELS_LOADED,
        'timestamp': datetime.now().isoformat()
    }
    return status

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("result.html", 
                         prediction="❌ Page Not Found",
                         error="The requested page does not exist."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template("result.html", 
                         prediction="❌ Internal Server Error",
                         error="Something went wrong on our end. Please try again later."), 500

if __name__ == '__main__':
    if not MODELS_LOADED:
        logger.error("Cannot start application: Models failed to load")
        exit(1)
    
    logger.info("Starting Loan Eligibility Prediction App...")
    app.run(debug=True, host='0.0.0.0', port=5000)