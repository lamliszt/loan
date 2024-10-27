from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Tải lại các thành phần cần thiết
feature_columns = load("feature_columns.joblib")
scaler = load("scaler.joblib")
clf_tree = load("decision_tree_model.joblib")
clf_rf = load("random_forest_model.joblib")
clf_lr = load("logistic_model.joblib")

# Hàm xử lý dữ liệu phân loại thành số
def preprocess_input(data):
    to_numeric = {
        'Male': 1, 'Female': 2,
        'Yes': 1, 'No': 2,
        'Graduate': 1, 'Not Graduate': 2,
        'Urban': 3, 'Semiurban': 2, 'Rural': 1,
        'Y': 1, 'N': 0,
        '3+': 3
    }
    data = data.replace(to_numeric)
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form và xác thực dữ liệu số
            data = {
                'Gender': request.form.get('Gender'),
                'Married': request.form.get('Married'),
                'Dependents': request.form.get('Dependents'),
                'Education': request.form.get('Education'),
                'Self_Employed': request.form.get('Self_Employed'),
                'ApplicantIncome': float(request.form.get('ApplicantIncome')),
                'CoapplicantIncome': float(request.form.get('CoapplicantIncome')),
                'LoanAmount': float(request.form.get('LoanAmount')),
                'Loan_Amount_Term': float(request.form.get('Loan_Amount_Term')),
                'Credit_History': float(request.form.get('Credit_History')),
                'Property_Area': request.form.get('Property_Area')
            }
        except ValueError:
            return render_template('index.html', error="Vui lòng nhập đúng định dạng cho các trường số.")

        # Xử lý và chuẩn bị dữ liệu để dự đoán
        input_df = pd.DataFrame([data])
        input_df = preprocess_input(input_df)
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        # Dự đoán với cả ba mô hình
        prediction_tree = 'Approved' if clf_tree.predict(input_scaled)[0] == 1 else 'Rejected'
        prediction_rf = 'Approved' if clf_rf.predict(input_scaled)[0] == 1 else 'Rejected'
        prediction_lr = 'Approved' if clf_lr.predict(input_scaled)[0] == 1 else 'Rejected'

        # Hiển thị kết quả trên giao diện
        return render_template(
            'index.html',
            prediction_tree=prediction_tree,
            prediction_rf=prediction_rf,
            prediction_lr=prediction_lr
        )

if __name__ == '__main__':
    app.run(debug=True)
