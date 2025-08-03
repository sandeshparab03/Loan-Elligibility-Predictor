"""
MongoDB Query Examples for Loan Prediction Application
Use these examples to interact with your loan prediction data
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from bson import ObjectId
import pandas as pd

# MongoDB Configuration
MONGODB_URI = 'mongodb://localhost:27017/'
DATABASE_NAME = "LoanPredictionDB"
COLLECTION_NAME = "Predictions"

class LoanDataQueries:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
    
    def get_basic_stats(self):
        """Get basic statistics about the loan data"""
        print("📊 Basic Statistics:")
        print("-" * 40)
        
        total_applications = self.collection.count_documents({})
        eligible_count = self.collection.count_documents({"prediction.eligible": True})
        ineligible_count = self.collection.count_documents({"prediction.eligible": False})
        
        print(f"Total Applications: {total_applications}")
        print(f"Approved Loans: {eligible_count} ({eligible_count/total_applications*100:.1f}%)")
        print(f"Rejected Loans: {ineligible_count} ({ineligible_count/total_applications*100:.1f}%)")
        
        # Average confidence scores
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$prediction.confidence"},
                "max_confidence": {"$max": "$prediction.confidence"},
                "min_confidence": {"$min": "$prediction.confidence"}
            }}
        ]
        
        stats = list(self.collection.aggregate(pipeline))
        if stats:
            stat = stats[0]
            print(f"Average Confidence: {stat['avg_confidence']:.2%}")
            print(f"Max Confidence: {stat['max_confidence']:.2%}")
            print(f"Min Confidence: {stat['min_confidence']:.2%}")
    
    def get_recent_applications(self, days=7):
        """Get applications from the last N days"""
        print(f"\n📅 Applications from Last {days} Days:")
        print("-" * 40)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_apps = self.collection.find(
            {"timestamp": {"$gte": cutoff_date}}
        ).sort("timestamp", -1)
        
        count = 0
        for app in recent_apps:
            count += 1
            input_data = app['input_data']
            prediction = app['prediction']
            
            print(f"\n{count}. {input_data.get('FirstName', 'N/A')} {input_data.get('LastName', 'N/A')}")
            print(f"   Status: {prediction['status']}")
            print(f"   Income: ₹{input_data['ApplicantIncome']:,}")
            print(f"   Loan: ₹{input_data['LoanAmount']:,}")
            print(f"   Date: {app['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            
            if count >= 10:  # Limit to 10 recent applications
                break
        
        if count == 0:
            print("No applications found in the specified period.")
    
    def get_high_confidence_predictions(self, min_confidence=0.9):
        """Get predictions with high confidence"""
        print(f"\n🎯 High Confidence Predictions (>{min_confidence:.0%}):")
        print("-" * 40)
        
        high_confidence = self.collection.find(
            {"prediction.confidence": {"$gte": min_confidence}}
        ).sort("prediction.confidence", -1)
        
        count = 0
        for pred in high_confidence:
            count += 1
            input_data = pred['input_data']
            prediction = pred['prediction']
            
            print(f"\n{count}. {prediction['status']} - {prediction['confidence']:.2%}")
            print(f"   Name: {input_data.get('FirstName', 'N/A')} {input_data.get('LastName', 'N/A')}")
            print(f"   Income: ₹{input_data['ApplicantIncome']:,} + ₹{input_data['CoapplicantIncome']:,}")
            print(f"   Loan: ₹{input_data['LoanAmount']:,} for {input_data['Loan_Amount_Term']} months")
        
        if count == 0:
            print("No high confidence predictions found.")
    
    def get_income_analysis(self):
        """Analyze loan eligibility by income ranges"""
        print(f"\n💰 Income Range Analysis:")
        print("-" * 40)
        
        pipeline = [
            {
                "$addFields": {
                    "total_income": {
                        "$add": ["$input_data.ApplicantIncome", "$input_data.CoapplicantIncome"]
                    },
                    "income_range": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lt": [{"$add": ["$input_data.ApplicantIncome", "$input_data.CoapplicantIncome"]}, 3000]}, "then": "Low (<₹3,000)"},
                                {"case": {"$lt": [{"$add": ["$input_data.ApplicantIncome", "$input_data.CoapplicantIncome"]}, 6000]}, "then": "Medium (₹3,000-6,000)"},
                                {"case": {"$lt": [{"$add": ["$input_data.ApplicantIncome", "$input_data.CoapplicantIncome"]}, 10000]}, "then": "High (₹6,000-10,000)"},
                            ],
                            "default": "Very High (>₹10,000)"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$income_range",
                    "total_applications": {"$sum": 1},
                    "approved": {"$sum": {"$cond": ["$prediction.eligible", 1, 0]}},
                    "avg_confidence": {"$avg": "$prediction.confidence"}
                }
            },
            {
                "$addFields": {
                    "approval_rate": {"$divide": ["$approved", "$total_applications"]}
                }
            },
            {"$sort": {"total_applications": -1}}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        for result in results:
            print(f"\n{result['_id']}:")
            print(f"   Applications: {result['total_applications']}")
            print(f"   Approved: {result['approved']} ({result['approval_rate']:.1%})")
            print(f"   Avg Confidence: {result['avg_confidence']:.2%}")
    
    def get_property_area_analysis(self):
        """Analyze loan eligibility by property area"""
        print(f"\n🏘️  Property Area Analysis:")
        print("-" * 40)
        
        pipeline = [
            {
                "$group": {
                    "_id": "$input_data.Property_Area",
                    "total_applications": {"$sum": 1},
                    "approved": {"$sum": {"$cond": ["$prediction.eligible", 1, 0]}},
                    "avg_loan_amount": {"$avg": "$input_data.LoanAmount"},
                    "avg_confidence": {"$avg": "$prediction.confidence"}
                }
            },
            {
                "$addFields": {
                    "approval_rate": {"$divide": ["$approved", "$total_applications"]}
                }
            },
            {"$sort": {"total_applications": -1}}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        for result in results:
            print(f"\n{result['_id']}:")
            print(f"   Applications: {result['total_applications']}")
            print(f"   Approved: {result['approved']} ({result['approval_rate']:.1%})")
            print(f"   Avg Loan Amount: ₹{result['avg_loan_amount']:,.0f}")
            print(f"   Avg Confidence: {result['avg_confidence']:.2%}")
    
    def search_by_name(self, name):
        """Search applications by applicant name"""
        print(f"\n🔍 Search Results for '{name}':")
        print("-" * 40)
        
        # Case-insensitive search in FirstName or LastName
        query = {
            "$or": [
                {"input_data.FirstName": {"$regex": name, "$options": "i"}},
                {"input_data.LastName": {"$regex": name, "$options": "i"}}
            ]
        }
        
        results = self.collection.find(query).sort("timestamp", -1)
        
        count = 0
        for result in results:
            count += 1
            input_data = result['input_data']
            prediction = result['prediction']
            
            print(f"\n{count}. {input_data.get('FirstName', 'N/A')} {input_data.get('LastName', 'N/A')}")
            print(f"   Status: {prediction['status']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Total Income: ₹{input_data['ApplicantIncome'] + input_data['CoapplicantIncome']:,}")
            print(f"   Loan Amount: ₹{input_data['LoanAmount']:,}")
            print(f"   Date: {result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        if count == 0:
            print(f"No applications found for '{name}'")
    
    def get_model_performance(self):
        """Compare performance between different model types"""
        print(f"\n🤖 Model Performance Comparison:")
        print("-" * 40)
        
        pipeline = [
            {
                "$group": {
                    "_id": "$model_type",
                    "total_predictions": {"$sum": 1},
                    "approved": {"$sum": {"$cond": ["$prediction.eligible", 1, 0]}},
                    "avg_confidence": {"$avg": "$prediction.confidence"},
                    "high_confidence_predictions": {
                        "$sum": {"$cond": [{"$gte": ["$prediction.confidence", 0.9]}, 1, 0]}
                    }
                }
            },
            {
                "$addFields": {
                    "approval_rate": {"$divide": ["$approved", "$total_predictions"]},
                    "high_confidence_rate": {"$divide": ["$high_confidence_predictions", "$total_predictions"]}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        for result in results:
            print(f"\n{result['_id'].replace('_', ' ').title()}:")
            print(f"   Total Predictions: {result['total_predictions']}")
            print(f"   Approval Rate: {result['approval_rate']:.1%}")
            print(f"   Avg Confidence: {result['avg_confidence']:.2%}")
            print(f"   High Confidence Rate: {result['high_confidence_rate']:.1%}")
    
    def export_to_csv(self, filename="loan_predictions.csv"):
        """Export all data to CSV file"""
        print(f"\n📤 Exporting data to {filename}...")
        
        # Get all documents
        cursor = self.collection.find({})
        
        # Flatten the data for CSV export
        flattened_data = []
        for doc in cursor:
            flat_record = {}
            
            # Add input data fields
            input_data = doc.get('input_data', {})
            for key, value in input_data.items():
                flat_record[f'input_{key}'] = value
            
            # Add prediction fields
            prediction = doc.get('prediction', {})
            for key, value in prediction.items():
                flat_record[f'prediction_{key}'] = value
            
            # Add metadata
            flat_record['model_type'] = doc.get('model_type')
            flat_record['timestamp'] = doc.get('timestamp')
            
            flattened_data.append(flat_record)
        
        # Create DataFrame and export
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        
        print(f"✅ Exported {len(flattened_data)} records to {filename}")
    
    def close_connection(self):
        """Close MongoDB connection"""
        self.client.close()

def main():
    """Main function to demonstrate queries"""
    print("🔍 MongoDB Query Examples for Loan Prediction Data")
    print("=" * 60)
    
    try:
        # Initialize query handler
        queries = LoanDataQueries()
        
        # Run various analyses
        queries.get_basic_stats()
        queries.get_recent_applications(days=7)
        queries.get_high_confidence_predictions(min_confidence=0.85)
        queries.get_income_analysis()
        queries.get_property_area_analysis()
        queries.get_model_performance()
        
        # Interactive search
        print(f"\n" + "="*60)
        search_name = input("Enter a name to search (or press Enter to skip): ").strip()
        if search_name:
            queries.search_by_name(search_name)
        
        # Export option
        export_choice = input("\nExport data to CSV? (y/N): ").strip().lower()
        if export_choice in ['y', 'yes']:
            queries.export_to_csv()
        
        # Close connection
        queries.close_connection()
        print("\n✅ Query session completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure MongoDB is running and contains loan prediction data.")

if __name__ == "__main__":
    main()