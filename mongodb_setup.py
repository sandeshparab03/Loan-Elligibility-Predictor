"""
MongoDB Data Setup Script for Loan Prediction Application
Run this script to populate your local MongoDB with sample prediction data
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import random
import json
import os
from bson import ObjectId

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = "LoanPredictionDB"
COLLECTION_NAME = "Predictions"

class MongoDBSetup:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(MONGODB_URI)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[DATABASE_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print(f"✅ Connected to MongoDB at {MONGODB_URI}")
            print(f"📊 Database: {DATABASE_NAME}")
            print(f"📦 Collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            print("Make sure MongoDB is running locally on port 27017")
            raise

    def clear_collection(self):
        """Clear existing data in the collection"""
        try:
            result = self.collection.delete_many({})
            print(f"🗑️  Cleared {result.deleted_count} existing records")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            raise

    def generate_sample_data(self, num_records=50):
        """Generate sample loan prediction data"""
        print(f"🔄 Generating {num_records} sample records...")
        
        # Sample data options
        genders = ['Male', 'Female']
        married_status = ['Yes', 'No']
        dependents = ['0', '1', '2', '3+']
        education = ['Graduate', 'Not Graduate']
        self_employed = ['Yes', 'No']
        property_areas = ['Urban', 'Semiurban', 'Rural']
        credit_histories = [0.0, 1.0]
        
        # Common Indian first and last names
        male_names = ['Raj', 'Amit', 'Suresh', 'Rahul', 'Vikram', 'Anil', 'Deepak', 'Sanjay', 'Ravi', 'Manoj']
        female_names = ['Priya', 'Sunita', 'Meera', 'Kavita', 'Anita', 'Rekha', 'Pooja', 'Neha', 'Ritu', 'Sita']
        last_names = ['Sharma', 'Patel', 'Kumar', 'Singh', 'Gupta', 'Verma', 'Agarwal', 'Jain', 'Yadav', 'Mishra']
        
        sample_records = []
        
        for i in range(num_records):
            # Generate random input data
            gender = random.choice(genders)
            first_name = random.choice(male_names if gender == 'Male' else female_names)
            last_name = random.choice(last_names)
            
            # Generate realistic income ranges
            applicant_income = random.randint(2000, 15000)
            coapplicant_income = random.randint(0, 8000) if random.choice([True, False]) else 0
            loan_amount = random.randint(50, 500)
            loan_term = random.choice([120, 180, 240, 300, 360, 480])
            
            input_data = {
                'FirstName': first_name,
                'LastName': last_name,
                'Gender': gender,
                'Married': random.choice(married_status),
                'Dependents': random.choice(dependents),
                'Education': random.choice(education),
                'Self_Employed': random.choice(self_employed),
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': random.choice(credit_histories),
                'Property_Area': random.choice(property_areas)
            }
            
            # Generate prediction result based on realistic criteria
            total_income = applicant_income + coapplicant_income
            emi = loan_amount / loan_term * 1000  # Convert to monthly EMI
            income_ratio = emi / total_income if total_income > 0 else 1
            
            # Simple eligibility logic (adjust as needed)
            eligible = (
                input_data['Credit_History'] == 1.0 and 
                income_ratio < 0.5 and 
                total_income > 3000
            )
            
            confidence = random.uniform(0.6, 0.95) if eligible else random.uniform(0.55, 0.85)
            probability_eligible = confidence if eligible else (1 - confidence)
            
            prediction_result = {
                'eligible': eligible,
                'status': "✅ Eligible" if eligible else "❌ Ineligible",
                'confidence': confidence,
                'probability_eligible': probability_eligible
            }
            
            # Create timestamp within last 30 days
            days_ago = random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago, 
                                                  hours=random.randint(0, 23),
                                                  minutes=random.randint(0, 59))
            
            record = {
                "input_data": input_data,
                "prediction": prediction_result,
                "model_type": random.choice(["native_xgboost", "sklearn_xgboost"]),
                "timestamp": timestamp
            }
            
            sample_records.append(record)
        
        return sample_records

    def insert_sample_data(self, records):
        """Insert sample data into MongoDB"""
        try:
            result = self.collection.insert_many(records)
            print(f"✅ Inserted {len(result.inserted_ids)} records successfully")
            return result.inserted_ids
        except Exception as e:
            print(f"❌ Error inserting data: {e}")
            raise

    def create_indexes(self):
        """Create useful indexes for the collection"""
        try:
            # Index on timestamp for time-based queries
            self.collection.create_index("timestamp")
            
            # Index on prediction status
            self.collection.create_index("prediction.status")
            
            # Index on model type
            self.collection.create_index("model_type")
            
            # Compound index for filtering
            self.collection.create_index([
                ("prediction.eligible", 1),
                ("timestamp", -1)
            ])
            
            print("📊 Created database indexes")
        except Exception as e:
            print(f"⚠️  Warning: Could not create indexes - {e}")

    def verify_data(self):
        """Verify the inserted data"""
        try:
            total_count = self.collection.count_documents({})
            eligible_count = self.collection.count_documents({"prediction.eligible": True})
            ineligible_count = self.collection.count_documents({"prediction.eligible": False})
            
            print(f"\n📈 Database Statistics:")
            print(f"   Total Records: {total_count}")
            print(f"   Eligible Loans: {eligible_count}")
            print(f"   Ineligible Loans: {ineligible_count}")
            
            # Show sample records
            print(f"\n📝 Sample Records:")
            sample_records = list(self.collection.find().limit(3))
            for i, record in enumerate(sample_records, 1):
                input_data = record['input_data']
                prediction = record['prediction']
                print(f"\n   Record {i}:")
                print(f"     Name: {input_data.get('FirstName', 'N/A')} {input_data.get('LastName', 'N/A')}")
                print(f"     Income: ₹{input_data['ApplicantIncome']:,}")
                print(f"     Loan Amount: ₹{input_data['LoanAmount']:,}")
                print(f"     Status: {prediction['status']}")
                print(f"     Confidence: {prediction['confidence']:.2%}")
                print(f"     Timestamp: {record['timestamp']}")
                
        except Exception as e:
            print(f"❌ Error verifying data: {e}")

    def add_custom_records(self):
        """Add some specific test records"""
        custom_records = [
            {
                "input_data": {
                    'FirstName': 'Test',
                    'LastName': 'User',
                    'Gender': 'Male',
                    'Married': 'Yes',
                    'Dependents': '1',
                    'Education': 'Graduate',
                    'Self_Employed': 'No',
                    'ApplicantIncome': 5000,
                    'CoapplicantIncome': 2000,
                    'LoanAmount': 150,
                    'Loan_Amount_Term': 360,
                    'Credit_History': 1.0,
                    'Property_Area': 'Urban'
                },
                "prediction": {
                    'eligible': True,
                    'status': "✅ Eligible",
                    'confidence': 0.87,
                    'probability_eligible': 0.87
                },
                "model_type": "native_xgboost",
                "timestamp": datetime.now()
            },
            {
                "input_data": {
                    'FirstName': 'Demo',
                    'LastName': 'Application',
                    'Gender': 'Female',
                    'Married': 'No',
                    'Dependents': '0',
                    'Education': 'Not Graduate',
                    'Self_Employed': 'Yes',
                    'ApplicantIncome': 2500,
                    'CoapplicantIncome': 0,
                    'LoanAmount': 200,
                    'Loan_Amount_Term': 240,
                    'Credit_History': 0.0,
                    'Property_Area': 'Rural'
                },
                "prediction": {
                    'eligible': False,
                    'status': "❌ Ineligible",
                    'confidence': 0.78,
                    'probability_eligible': 0.22
                },
                "model_type": "sklearn_xgboost",
                "timestamp": datetime.now() - timedelta(hours=2)
            }
        ]
        
        try:
            result = self.collection.insert_many(custom_records)
            print(f"✅ Added {len(result.inserted_ids)} custom test records")
        except Exception as e:
            print(f"❌ Error adding custom records: {e}")

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 Closed MongoDB connection")

def main():
    """Main function to set up MongoDB data"""
    print("🚀 MongoDB Data Setup for Loan Prediction Application")
    print("=" * 60)
    
    try:
        # Initialize MongoDB setup
        mongo_setup = MongoDBSetup()
        
        # Ask user if they want to clear existing data
        print(f"\n📊 Checking existing data...")
        existing_count = mongo_setup.collection.count_documents({})
        
        if existing_count > 0:
            print(f"Found {existing_count} existing records in the collection.")
            response = input("Do you want to clear existing data? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                mongo_setup.clear_collection()
        
        # Get number of records to generate
        try:
            num_records = int(input("\nHow many sample records to generate? (default: 50): ") or "50")
        except ValueError:
            num_records = 50
            print("Using default: 50 records")
        
        # Generate and insert sample data
        sample_records = mongo_setup.generate_sample_data(num_records)
        mongo_setup.insert_sample_data(sample_records)
        
        # Add custom test records
        mongo_setup.add_custom_records()
        
        # Create indexes
        mongo_setup.create_indexes()
        
        # Verify the data
        mongo_setup.verify_data()
        
        print(f"\n🎉 MongoDB setup completed successfully!")
        print(f"📍 Your Flask app can now access this data at: {MONGODB_URI}")
        print(f"📱 You can view the data using MongoDB Compass or mongo shell")
        
        # Close connection
        mongo_setup.close_connection()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please ensure MongoDB is running and accessible.")

if __name__ == "__main__":
    main()