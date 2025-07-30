# Employee Salary Prediction & Analysis 💼💰

A sleek web application that predicts employee salaries using advanced machine learning models and provides insightful analysis to help users make informed career decisions.

---

## Website Snapshots
![Home Page](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/Home%20Page.png)
![Home Page Result](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/Home%20Page%20Result.png)
![Sidemenu](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/Menubar%20Image.png)
![About Page](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/About.html.png)
![Recommendations](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/Recommendations.html.png)
![Trends](https://github.com/Jaykd33/Employee-Salary-Prediction/blob/main/Trends.html.png)


## Features ✨

- Predict salaries based on age, gender, department, job title, experience, education, and location.
- Displays model accuracy to build user trust.
- Interactive and premium UI with smooth navigation.
- Additional pages with recent salary trends, recommended studies, and career advice.
- Honest feedback and job profile suggestions.

---

## Tech Stack 🛠️

- Python, Flask for backend API
- Scikit-learn (Linear Regression), XGBoost for ML model training
- HTML, CSS, JavaScript for frontend
- Jinja2 templating engine

---

## Getting Started 🚀

1. **Clone the repo**

```bash
git clone <repo-url>
cd <repo-folder>
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model**

```bash
python salary_prediction_model.py
```

4. **Run the Flask app**

```bash
python app.py
```

5. **Open your browser**

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the app.

---

## Machine Learning Workflow 🧠

### 1. Data Preprocessing and Exploratory Data Analysis (EDA)

- Conducted in the `Project.ipynb` notebook.
- Loaded and cleaned the dataset (`Employers_data.csv`), handling missing values and irrelevant columns.
- Performed exploratory data analysis using pandas, matplotlib, seaborn, and plotly to visualize distributions and relationships between features such as Age, Salary, Gender, Location, and more.

### 2. Model Building and Training

- Implemented in `salary_prediction_model.py`.
- Loaded and preprocessed data by encoding categorical variables and splitting into training and test sets.
- Trained multiple regression models: Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
- Evaluated models using R2 score to determine the best performing model.
- The best model achieved an R2 score of approximately 0.95, indicating high accuracy in salary prediction.
- Saved the best model (`best_salary_model.pkl`) and its accuracy metrics (`model_accuracy.json`).

### 3. Integration with Web Application

- The Flask app (`app.py`) loads the best trained model and accuracy data.
- Provides a web UI (`templates/index.html`) where users input employee details to get salary predictions.
- Displays the best model's accuracy score on the UI to build user trust.
- Additional pages provide salary trends, study recommendations, and career advice.

---

## Usage 🧑‍💻

- Fill in employee details in the form.
- Get instant salary prediction with accuracy info.
- Explore trends, recommendations, and career advice via the sidebar menu.

---

## Contribution 🤝

Feel free to open issues or submit pull requests for improvements!

---


