# Loan Eligibility Predictor

A Flask web application that predicts loan eligibility using a machine learning model and stores prediction logs in MongoDB. Includes tools for querying and analyzing prediction data.

## Features
- User-friendly web form for loan eligibility prediction
- Machine learning model (XGBoost) for predictions
- Scaler and feature engineering for accurate results
- MongoDB integration for logging predictions
- Query and analysis scripts for data insights

## Project Structure
```
app.py                  # Main Flask application
train_model.py          # (If present) Model training or test script
model/
  LoanElligibilityPredictor.pkl  # Trained XGBoost model (or .json if using Booster.save_model)
  scaler.pkl            # Scaler for input features
  columns.pkl           # List of expected columns/features
static/
  styles.css            # CSS for the web app
  README.md             # Static folder info
templates/
  index.html            # Main form page
  result.html           # Prediction result page
mongodb_query_examples.py # MongoDB data analysis and export tool
requirements.txt        # Python dependencies
README.md               # Project documentation (this file)
```

## Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. MongoDB Setup
- Ensure MongoDB is running locally on `mongodb://localhost:27017/`.
- The app uses a database named `LoanPredictionDB` and a collection named `Predictions`.

### 3. Model Files
- Place `LoanElligibilityPredictor.pkl` (or `.json` if using XGBoost Booster), `scaler.pkl`, and `columns.pkl` in the `model/` directory.

### 4. Run the Flask App
```bash
python app.py
```
- Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5. Query and Analyze Data
- Use `mongodb_query_examples.py` to analyze and export prediction data:
```bash
python mongodb_query_examples.py
```

## Notes
- For best results, use the same versions of scikit-learn and XGBoost for both training and inference.
- If you see warnings about model serialization, see the comments in `train_model.py` and `app.py` for migration steps.
- Update the model and scaler files as needed if you retrain or update your ML pipeline.

## License
This project is for educational/demo purposes. Modify and use as needed.
