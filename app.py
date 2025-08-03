import warnings
# Suppress specific version warnings (temporary fix)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*sklearn.*version.*", category=UserWarning)

from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import os
import xgboost as xgb
from pymongo import MongoClient
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.expected_columns = None
        self.db_collection = None
        self.is_native_xgb = False
        self._load_models()
        self._setup_database()
    
    def _load_models(self):
        """Load ML models and preprocessors with version handling"""
        try:
            # Try to load XGBoost model in order of preference
            if os.path.exists('model/LoanElligibilityPredictor.json'):
                logger.info("Loading XGBoost model from native JSON format...")
                self.model = xgb.Booster()
                self.model.load_model('model/LoanElligibilityPredictor.json')
                self.is_native_xgb = True
            elif os.path.exists('model/LoanElligibilityPredictor_updated.pkl'):
                logger.info("Loading XGBoost model from updated pickle...")
                with open('model/LoanElligibilityPredictor_updated.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                self.is_native_xgb = False
            else:
                logger.info("Loading XGBoost model from original pickle (may show warnings)...")
                with open('model/LoanElligibilityPredictor.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                self.is_native_xgb = False
            
            # Load scaler (prefer updated version)
            scaler_path = 'model/scaler_updated.pkl' if os.path.exists('model/scaler_updated.pkl') else 'model/scaler.pkl'
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load columns (prefer updated version)
            columns_path = 'model/columns_updated.pkl' if os.path.exists('model/columns_updated.pkl') else 'model/columns.pkl'
            with open(columns_path, 'rb') as f:
                self.expected_columns = pickle.load(f)
            
            logger.info(f"Models loaded successfully (XGBoost: {'native' if self.is_native_xgb else 'sklearn-wrapper'})")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _setup_database(self):
        """Setup MongoDB connection"""
        try:
            client = MongoClient(MONGODB_URI)
            db = client["LoanPredictionDB"]
            self.db_collection = db["Predictions"]
            # Test connection
            client.admin.command('ping')
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Don't raise - app can work without DB
            self.db_collection = None
    
    def _validate_input(self, input_data):
        """Validate input data"""
        required_fields = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                          'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                          'Property_Area']
        
        # Remove non-model fields (like FirstName, LastName)
        model_data = {k: v for k, v in input_data.items() if k in required_fields + ['FirstName', 'LastName']}
        
        missing_fields = [field for field in required_fields if field not in model_data or str(model_data[field]).strip() == '']
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate numeric fields
        numeric_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History']
        
        for field in numeric_fields:
            try:
                value = float(model_data[field])
                if value < 0:
                    raise ValueError(f"{field.replace('_', ' ')} cannot be negative")
                if field in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'] and value == 0:
                    raise ValueError(f"{field.replace('_', ' ')} cannot be zero")
                model_data[field] = value
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field}: {model_data[field]}")
        
        # Additional validations
        if model_data['Credit_History'] not in [0.0, 1.0]:
            raise ValueError("Credit History must be 0 or 1")
        
        if model_data['Loan_Amount_Term'] < 12:
            raise ValueError("Loan Amount Term must be at least 12 months")
        
        return model_data
    
    def _preprocess_data(self, input_data):
        """Preprocess input data for prediction"""
        df = pd.DataFrame([input_data])
        
        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomePerLoan'] = df['TotalIncome'] / (df['LoanAmount'] + 1e-6)
        df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df)
        
        # Align columns with training data
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.expected_columns]
        
        # Scale features
        scaled_input = self.scaler.transform(df)
        
        return scaled_input
    
    def predict(self, input_data):
        """Make loan eligibility prediction"""
        # Validate input
        validated_data = self._validate_input(input_data.copy())
        
        # Preprocess
        processed_data = self._preprocess_data(validated_data)
        
        # Make prediction based on model type
        if self.is_native_xgb:
            # Native XGBoost model
            dtest = xgb.DMatrix(processed_data)
            prediction_proba = self.model.predict(dtest)[0]
            prediction = 1 if prediction_proba > 0.5 else 0
            probability = [1 - prediction_proba, prediction_proba]
        else:
            # Scikit-learn style XGBoost model
            prediction = self.model.predict(processed_data)[0]
            try:
                probability = self.model.predict_proba(processed_data)[0]
            except:
                # Fallback if predict_proba is not available
                prediction_proba = float(prediction)
                probability = [1 - prediction_proba, prediction_proba]
        
        result = {
            'eligible': bool(prediction),
            'status': "✅ Eligible" if prediction == 1 else "❌ Ineligible",
            'confidence': float(max(probability)),
            'probability_eligible': float(probability[1]) if len(probability) > 1 else float(probability[0])
        }
        
        # Log prediction
        self._log_prediction(input_data, result)
        
        return result
    
    def _log_prediction(self, input_data, result):
        """Log prediction to database"""
        if self.db_collection is None:
            logger.warning("Database not available, skipping logging")
            return
        
        try:
            log_entry = {
                "input_data": input_data,
                "prediction": result,
                "model_type": "native_xgboost" if self.is_native_xgb else "sklearn_xgboost",
                "timestamp": datetime.now()
            }
            self.db_collection.insert_one(log_entry)
            logger.info("Prediction logged successfully")
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

# Initialize predictor
try:
    predictor = LoanPredictor()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/test')
def test():
    return "Flask app is working! ✅"

@app.route('/predict', methods=['POST'])
@app.route('/predict_form', methods=['POST'])
def predict_form():
    logger.info(f"Received prediction request from {request.remote_addr}")
    logger.info(f"Form data keys: {list(request.form.keys())}")
    
    if predictor is None:
        return render_template("result.html", 
                             prediction="❌ Service temporarily unavailable",
                             error="Model not loaded")
    
    try:
        # Extract form data
        input_data = {k: request.form[k] for k in request.form}
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Calculate additional financial info
        total_income = float(input_data['ApplicantIncome']) + float(input_data['CoapplicantIncome'])
        emi = float(input_data['LoanAmount']) / float(input_data['Loan_Amount_Term'])
        
        return render_template("result.html", 
                             prediction=result['status'],
                             confidence=f"{result['confidence']:.2%}",
                             probability=f"{result['probability_eligible']:.2%}",
                             total_income=f"₹{total_income:,.0f}",
                             emi=f"₹{emi:,.0f}",
                             input_data=input_data)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return render_template("result.html", 
                             prediction="❌ Invalid input",
                             error=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("result.html", 
                             prediction="❌ Prediction failed",
                             error="Please try again later")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if predictor is None:
        return jsonify({
            'error': 'Service temporarily unavailable',
            'status': 'error'
        }), 503
    
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                'error': 'No input data provided',
                'status': 'error'
            }), 400
        
        result = predictor.predict(input_data)
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'model_info': {
                'type': 'native_xgboost' if predictor.is_native_xgb else 'sklearn_xgboost',
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if predictor else 'unhealthy',
        'model_loaded': predictor is not None,
        'model_type': 'native_xgboost' if predictor and predictor.is_native_xgb else 'sklearn_xgboost',
        'database_connected': predictor.db_collection is not None if predictor else False,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status), 200 if predictor else 503

@app.route('/model-info')
def model_info():
    """Get information about loaded models"""
    if not predictor:
        return jsonify({'error': 'Models not loaded'}), 503
    
    info = {
        'xgboost_model': 'native_json' if predictor.is_native_xgb else 'sklearn_pickle',
        'scaler_file': 'scaler_updated.pkl' if os.path.exists('model/scaler_updated.pkl') else 'scaler.pkl',
        'columns_file': 'columns_updated.pkl' if os.path.exists('model/columns_updated.pkl') else 'columns.pkl',
        'database_status': 'connected' if predictor.db_collection is not None else 'disconnected',
        'available_endpoints': ['/predict_form', '/api/predict', '/health', '/model-info']
    }
    
    return jsonify(info)

@app.errorhandler(404)
def not_found(error):
    return render_template("result.html", 
                         prediction="❌ Page not found",
                         error="The requested page does not exist"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("result.html", 
                         prediction="❌ Internal server error",
                         error="Something went wrong. Please try again later"), 500

if __name__ == '__main__':
    if not predictor:
        logger.error("Cannot start application: Models failed to load")
        exit(1)
    
    logger.info("Starting Loan Eligibility Prediction App...")
    logger.info(f"Model type: {'Native XGBoost' if predictor.is_native_xgb else 'Sklearn XGBoost'}")
    logger.info(f"Database: {'Connected' if predictor.db_collection is not None else 'Disconnected'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)