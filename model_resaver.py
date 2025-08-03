"""
Script to re-save models with current library versions
Run this once to fix version compatibility issues
"""

import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import os

def resave_models():
    """Load and re-save models with current library versions"""
    
    print("🔄 Loading existing models...")
    
    # Suppress warnings during loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            # Load existing models
            with open('model/LoanElligibilityPredictor.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            with open('model/columns.pkl', 'rb') as f:
                expected_columns = pickle.load(f)
            
            print("✅ Models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model file not found - {e}")
            print("Make sure all model files exist in the 'model/' directory:")
            print("  - LoanElligibilityPredictor.pkl")
            print("  - scaler.pkl") 
            print("  - columns.pkl")
            return False
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    # Create backup directory
    backup_dir = 'model/backup'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"📁 Created backup directory: {backup_dir}")
    
    # Backup original files
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        shutil.copy2('model/LoanElligibilityPredictor.pkl', f'{backup_dir}/LoanElligibilityPredictor_{timestamp}.pkl')
        shutil.copy2('model/scaler.pkl', f'{backup_dir}/scaler_{timestamp}.pkl')
        shutil.copy2('model/columns.pkl', f'{backup_dir}/columns_{timestamp}.pkl')
        print(f"💾 Original models backed up with timestamp: {timestamp}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create backup - {e}")
    
    # For XGBoost, save using the native format (recommended)
    print("🔄 Saving XGBoost model in native JSON format...")
    try:
        if hasattr(model, 'save_model'):
            # Native XGBoost model
            model.save_model('model/LoanElligibilityPredictor.json')
        else:
            # Sklearn wrapper - extract the booster
            if hasattr(model, '_Booster'):
                model._Booster.save_model('model/LoanElligibilityPredictor.json')
            elif hasattr(model, 'get_booster'):
                model.get_booster().save_model('model/LoanElligibilityPredictor.json')
            else:
                print("⚠️  Could not extract native XGBoost model, keeping pickle format")
                # Re-save pickle with current version
                with open('model/LoanElligibilityPredictor_updated.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        print("✅ XGBoost model saved in native format!")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save native XGBoost format - {e}")
        print("🔄 Re-saving as pickle with current version...")
        with open('model/LoanElligibilityPredictor_updated.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Re-save scaler with current sklearn version
    print("🔄 Re-saving scaler with current sklearn version...")
    try:
        with open('model/scaler_updated.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Scaler re-saved successfully!")
    except Exception as e:
        print(f"❌ Error saving scaler: {e}")
        return False
    
    # Columns file should be fine, but re-save anyway
    print("🔄 Re-saving columns...")
    try:
        with open('model/columns_updated.pkl', 'wb') as f:
            pickle.dump(expected_columns, f)
        print("✅ Columns re-saved successfully!")
    except Exception as e:
        print(f"❌ Error saving columns: {e}")
        return False
    
    print("\n🎉 All models re-saved successfully!")
    return True

def test_models():
    """Test that the re-saved models work correctly"""
    print("\n🧪 Testing re-saved models...")
    
    try:
        # Try to load XGBoost model (native format first)
        if os.path.exists('model/LoanElligibilityPredictor.json'):
            print("📊 Loading XGBoost model from native JSON format...")
            model = xgb.Booster()
            model.load_model('model/LoanElligibilityPredictor.json')
            is_native = True
        elif os.path.exists('model/LoanElligibilityPredictor_updated.pkl'):
            print("📊 Loading XGBoost model from updated pickle...")
            with open('model/LoanElligibilityPredictor_updated.pkl', 'rb') as f:
                model = pickle.load(f)
            is_native = False
        else:
            print("❌ No updated model files found!")
            return False
        
        # Load updated scaler
        with open('model/scaler_updated.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load columns
        with open('model/columns_updated.pkl', 'rb') as f:
            expected_columns = pickle.load(f)
        
        print("✅ All models loaded without warnings!")
        
    except Exception as e:
        print(f"❌ Error loading updated models: {e}")
        return False
    
    # Create a test prediction
    print("🧪 Running test prediction...")
    test_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000.0,
        'CoapplicantIncome': 2000.0,
        'LoanAmount': 150.0,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    try:
        # Preprocess test data
        df = pd.DataFrame([test_data])
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomePerLoan'] = df['TotalIncome'] / (df['LoanAmount'] + 1e-6)
        df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
        
        df = pd.get_dummies(df)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]
        
        scaled_input = scaler.transform(df)
        
        # Make prediction based on model type
        if is_native:
            # Convert to DMatrix for native XGBoost
            dtest = xgb.DMatrix(scaled_input)
            prediction_proba = model.predict(dtest)[0]
            prediction = 1 if prediction_proba > 0.5 else 0
        else:
            # Sklearn wrapper
            prediction = model.predict(scaled_input)[0]
            try:
                prediction_proba = model.predict_proba(scaled_input)[0][1]
            except:
                prediction_proba = float(prediction)
        
        result = "✅ Eligible" if prediction == 1 else "❌ Ineligible"
        print(f"🎯 Test prediction: {result} (confidence: {prediction_proba:.4f})")
        print("✅ Models are working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test prediction: {e}")
        return False

def update_app_files():
    """Update app.py to use the new model files"""
    print("\n📝 Next steps:")
    print("1. Your original models have been backed up in model/backup/")
    
    if os.path.exists('model/LoanElligibilityPredictor.json'):
        print("2. Update your app.py to use the native XGBoost format")
        print("3. The updated files are:")
        print("   - LoanElligibilityPredictor.json (native XGBoost)")
        print("   - scaler_updated.pkl (current sklearn version)")
        print("   - columns_updated.pkl (current version)")
    else:
        print("2. The updated files are:")
        print("   - LoanElligibilityPredictor_updated.pkl (current XGBoost)")
        print("   - scaler_updated.pkl (current sklearn version)")
        print("   - columns_updated.pkl (current version)")
    
    print("4. Test your Flask app to ensure everything works")
    print("5. If successful, you can replace the original files")

if __name__ == "__main__":
    print("🚀 Model Re-saver Script")
    print("=" * 50)
    
    try:
        # Check if model directory exists
        if not os.path.exists('model'):
            print("❌ Error: 'model' directory not found!")
            print("Make sure you're running this script from the correct directory.")
            exit(1)
        
        success = resave_models()
        if success:
            test_success = test_models()
            if test_success:
                update_app_files()
                print("\n🎉 Model re-saving completed successfully!")
                print("You can now run your Flask app without version warnings.")
            else:
                print("\n⚠️  Models were re-saved but testing failed.")
                print("Please check the error messages above.")
        else:
            print("\n❌ Model re-saving failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your model files and try again.")