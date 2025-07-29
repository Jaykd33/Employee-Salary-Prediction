from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import json
import traceback

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_salary_model.pkl')

# Load model accuracy
try:
    with open('model_accuracy.json', 'r') as f:
        accuracy_data = json.load(f)
except Exception:
    accuracy_data = {'best_model': 'Unknown', 'r2_score': None}

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    required_fields = ['Age', 'Gender', 'Department', 'Job_Title', 'Experience_Years', 'Education_Level', 'Location']

    # Validate all required fields are present and non-empty
    missing_fields = [field for field in required_fields if field not in data or data[field].strip() == '']
    if missing_fields:
        error_message = f"Missing or empty fields: {', '.join(missing_fields)}"
        return render_template('index.html', error=error_message, input_data=data, accuracy=accuracy_data)

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        # Convert numeric columns to appropriate types
        input_df['Age'] = input_df['Age'].astype(int)
        input_df['Experience_Years'] = input_df['Experience_Years'].astype(int)
        # Predict salary
        prediction = model.predict(input_df)[0]
    except Exception as e:
        tb = traceback.format_exc()
        error_message = f"Error during prediction: {str(e)}\nTraceback:\n{tb}"
        return render_template('index.html', error=error_message, input_data=data, accuracy=accuracy_data)

    # Additional analysis data (static placeholders for now)
    analysis = {
        'recent_trends': 'Salaries in tech and finance sectors are rising steadily.',
        'recommended_studies': 'Consider courses in Data Science, AI, and Cloud Computing.',
        'job_profiles': ['Data Scientist', 'Software Engineer', 'Financial Analyst', 'Product Manager'],
        'top_companies': ['Google', 'Microsoft', 'Amazon', 'Goldman Sachs'],
        'honest_feedback': 'Your profile looks promising! Keep upgrading your skills.'
    }
    return render_template('index.html', prediction=round(prediction, 2), input_data=data, accuracy=accuracy_data, analysis=analysis)

@app.route('/about')
def about():
    return render_template('about.html', accuracy=accuracy_data)

@app.route('/trends')
def trends():
    return render_template('trends.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

if __name__ == '__main__':
    app.run(debug=True)
