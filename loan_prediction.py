# Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Đọc dữ liệu từ file CSV
print("Đang đọc dữ liệu...")
Tr_data = pd.read_csv('loan_data.csv')
print("Dữ liệu Loan Train:")
print(Tr_data.head())

# Xử lý dữ liệu: điền giá trị thiếu và chuyển đổi dữ liệu phân loại thành số
print("\nĐiền các giá trị thiếu...")
null_cols = ['Credit_History', 'Self_Employed', 'Dependents', 'Loan_Amount_Term', 'Gender']
for col in null_cols:
    Tr_data[col].fillna(Tr_data[col].dropna().mode()[0], inplace=True)

# Chuyển đổi các cột phân loại thành số
to_numeric = {
    'Male': 1, 'Female': 2,
    'Yes': 1, 'No': 2,
    'Graduate': 1, 'Not Graduate': 2,
    'Urban': 3, 'Semiurban': 2, 'Rural': 1,
    'Y': 1, 'N': 0,
    '3+': 3
}
Tr_data = Tr_data.applymap(lambda label: to_numeric.get(label) if label in to_numeric else label)

# Lưu cột Loan_ID và loại bỏ cột này khỏi dữ liệu chính
loan_ids = Tr_data['Loan_ID'].copy()
Tr_data.drop('Loan_ID', axis=1, inplace=True)

# Chia dữ liệu thành đặc trưng (X) và nhãn (y)
X_data = Tr_data.drop('Loan_Status', axis=1)
y_data = Tr_data['Loan_Status']

# Lưu tên các cột của X_train để sử dụng lại trong các dự đoán
feature_columns = X_data.columns.to_list()
dump(feature_columns, 'feature_columns.joblib')

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu và lưu scaler để sử dụng lại
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dump(scaler, 'scaler.joblib')

# Huấn luyện và lưu mô hình Decision Tree
print("\nHuấn luyện mô hình Decision Tree...")
clf_tree = DecisionTreeClassifier(max_depth=7, min_samples_split=10)
clf_tree.fit(X_train_scaled, y_train)
dump(clf_tree, 'decision_tree_model.joblib')

# Huấn luyện và lưu mô hình Random Forest
print("\nHuấn luyện mô hình Random Forest...")
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5)
clf_rf.fit(X_train_scaled, y_train)
dump(clf_rf, 'random_forest_model.joblib')

# Huấn luyện và lưu mô hình Logistic Regression
print("\nHuấn luyện mô hình Logistic Regression...")
clf_lr = LogisticRegression(max_iter=300, C=0.1, penalty='l2')
clf_lr.fit(X_train_scaled, y_train)
dump(clf_lr, 'logistic_model.joblib')

# Dự đoán trên tập kiểm tra và lưu độ chính xác của từng mô hình
y_test_pred_tree = clf_tree.predict(X_test_scaled)
y_test_pred_rf = clf_rf.predict(X_test_scaled)
y_test_pred_lr = clf_lr.predict(X_test_scaled)

print("Độ chính xác của Decision Tree trên tập kiểm tra: %.2f%%" % (accuracy_score(y_test, y_test_pred_tree) * 100))
print("Độ chính xác của Random Forest trên tập kiểm tra: %.2f%%" % (accuracy_score(y_test, y_test_pred_rf) * 100))
print("Độ chính xác của Logistic Regression trên tập kiểm tra: %.2f%%" % (accuracy_score(y_test, y_test_pred_lr) * 100))

# Hàm dùng để dự đoán từ dữ liệu đầu vào mới
def predict_loan_status(input_data):
    # Tải scaler, feature_columns và cả ba mô hình từ file
    scaler = load('scaler.joblib')
    feature_columns = load('feature_columns.joblib')
    model_tree = load('decision_tree_model.joblib')
    model_rf = load('random_forest_model.joblib')
    model_lr = load('logistic_model.joblib')
    
    # Sắp xếp cột của input_data theo thứ tự `feature_columns`
    input_data = input_data[feature_columns]
    
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_data)
    
    # Dự đoán với từng mô hình
    prediction_tree = model_tree.predict(input_scaled)[0]
    prediction_rf = model_rf.predict(input_scaled)[0]
    prediction_lr = model_lr.predict(input_scaled)[0]
    
    # Trả về kết quả của cả ba mô hình
    return {
        'Decision Tree': 'Approved' if prediction_tree == 1 else 'Rejected',
        'Random Forest': 'Approved' if prediction_rf == 1 else 'Rejected',
        'Logistic Regression': 'Approved' if prediction_lr == 1 else 'Rejected'
    }

# Dự đoán và lưu kết quả vào file CSV
print("\nDự đoán và lưu kết quả vào file CSV...")
all_predictions_tree = clf_tree.predict(X_data)
all_predictions_rf = clf_rf.predict(X_data)
all_predictions_lr = clf_lr.predict(X_data)

Tr_data['Decision_Tree'] = ['Approved' if pred == 1 else 'Rejected' for pred in all_predictions_tree]
Tr_data['Random_Forest'] = ['Approved' if pred == 1 else 'Rejected' for pred in all_predictions_rf]
Tr_data['Logistic_Regression'] = ['Approved' if pred == 1 else 'Rejected' for pred in all_predictions_lr]
Tr_data['Loan_ID'] = loan_ids

Tr_data[['Loan_ID', 'Decision_Tree', 'Random_Forest', 'Logistic_Regression']].to_csv('loan_predictions.csv', index=False)
print("Kết quả dự đoán đã được lưu vào loan_predictions.csv")
