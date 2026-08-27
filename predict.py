import joblib
import pandas as pd
import numpy as np
import sys
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
CSV_OUT = os.path.join(os.path.dirname(__file__), 'hasil_prediksi.csv')

COLS_NEEDED = [
    'Marital_status','Application_mode','Application_order','Course',
    'Daytime_evening_attendance','Previous_qualification','Previous_qualification_grade',
    'Nacionality','Mothers_qualification','Fathers_qualification',
    'Mothers_occupation','Fathers_occupation','Admission_grade',
    'Displaced','Educational_special_needs','Debtor','Tuition_fees_up_to_date',
    'Gender','Scholarship_holder','Age_at_enrollment','International',
    'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade','Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited','Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations','Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade','Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate','Inflation_rate','GDP'
]

def predict_file(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    # ensure all cols present
    for c in COLS_NEEDED:
        if c not in df.columns:
            df[c] = 0
    df = df[COLS_NEEDED]
    model = joblib.load(MODEL_PATH)
    pred = model.predict(df)
    prob = model.predict_proba(df)[:,1]
    df['prediksi'] = np.where(pred==1,'Dropout','Graduate')
    df['prob_dropout'] = prob
    df.to_csv(CSV_OUT, index=False, sep=';')
    print(f'Hasil disimpan ke {CSV_OUT}')
    print(df[['prediksi','prob_dropout']].head())

def predict_single():
    print("Masukkan data siswa untuk prediksi dropout:\n")
    data = {}
    data['Marital_status'] = int(input("Marital_status (1=single,2=married,3=widow,4=divorced,5=union,6=separated): "))
    data['Application_mode'] = int(input("Application_mode (1-18): "))
    data['Application_order'] = int(input("Application_order: "))
    data['Course'] = int(input("Course code (contoh 171,9254): "))
    data['Daytime_evening_attendance'] = int(input("Daytime_evening_attendance (1=daytime,0=evening): "))
    data['Previous_qualification'] = int(input("Previous_qualification: "))
    data['Previous_qualification_grade'] = float(input("Previous_qualification_grade: "))
    data['Nacionality'] = int(input("Nacionality: "))
    data['Mothers_qualification'] = int(input("Mothers_qualification: "))
    data['Fathers_qualification'] = int(input("Fathers_qualification: "))
    data['Mothers_occupation'] = int(input("Mothers_occupation: "))
    data['Fathers_occupation'] = int(input("Fathers_occupation: "))
    data['Admission_grade'] = float(input("Admission_grade: "))
    data['Displaced'] = int(input("Displaced (1=yes,0=no): "))
    data['Educational_special_needs'] = int(input("Educational_special_needs (1=yes,0=no): "))
    data['Debtor'] = int(input("Debtor (1=yes,0=no): "))
    data['Tuition_fees_up_to_date'] = int(input("Tuition_fees_up_to_date (1=yes,0=no): "))
    data['Gender'] = int(input("Gender (1=male,0=female): "))
    data['Scholarship_holder'] = int(input("Scholarship_holder (1=yes,0=no): "))
    data['Age_at_enrollment'] = int(input("Age_at_enrollment: "))
    data['International'] = int(input("International (1=yes,0=no): "))
    data['Curricular_units_1st_sem_credited'] = int(input("Curricular_units_1st_sem_credited: "))
    data['Curricular_units_1st_sem_enrolled'] = int(input("Curricular_units_1st_sem_enrolled: "))
    data['Curricular_units_1st_sem_evaluations'] = int(input("Curricular_units_1st_sem_evaluations: "))
    data['Curricular_units_1st_sem_approved'] = int(input("Curricular_units_1st_sem_approved: "))
    data['Curricular_units_1st_sem_grade'] = float(input("Curricular_units_1st_sem_grade: "))
    data['Curricular_units_1st_sem_without_evaluations'] = int(input("Curricular_units_1st_sem_without_evaluations: "))
    data['Curricular_units_2nd_sem_credited'] = int(input("Curricular_units_2nd_sem_credited: "))
    data['Curricular_units_2nd_sem_enrolled'] = int(input("Curricular_units_2nd_sem_enrolled: "))
    data['Curricular_units_2nd_sem_evaluations'] = int(input("Curricular_units_2nd_sem_evaluations: "))
    data['Curricular_units_2nd_sem_approved'] = int(input("Curricular_units_2nd_sem_approved: "))
    data['Curricular_units_2nd_sem_grade'] = float(input("Curricular_units_2nd_sem_grade: "))
    data['Curricular_units_2nd_sem_without_evaluations'] = int(input("Curricular_units_2nd_sem_without_evaluations: "))
    data['Unemployment_rate'] = float(input("Unemployment_rate: "))
    data['Inflation_rate'] = float(input("Inflation_rate: "))
    data['GDP'] = float(input("GDP: "))

    df = pd.DataFrame([data])
    model = joblib.load(MODEL_PATH)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0,1]
    status = 'Dropout' if pred==1 else 'Graduate'
    print(f"\nPrediksi: {status}")
    print(f"Probabilitas Dropout: {prob:.2%}")
    return status, prob

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict_file(sys.argv[1])
    else:
        predict_single()
