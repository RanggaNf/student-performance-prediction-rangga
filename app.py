import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Jaya Jaya Institut - Dropout Prediction", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('students_performance.csv', sep=';')
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join('model', 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# Identity
st.sidebar.markdown("""
### Identitas
- **Nama:** Mohammad Rangga Nugraha Firmansyah  
- **Email:** mohammad.rangga.n.f@gmail.com  
- **Id Dicoding:** rangganf
""")

# Sidebar navigation
page = st.sidebar.radio("Navigasi", ["Dashboard", "Predict"])

# Dashboard page
if page == "Dashboard":
    st.title("Jaya Jaya Institut - Dashboard Performa Siswa")
    st.markdown("Dashboard ini menampilkan berbagai faktor yang mempengaruhi performa dan risiko dropout siswa.")

    # KPIs
    total = len(df)
    dropout = (df['Status']=='Dropout').sum()
    enrolled = (df['Status']=='Enrolled').sum()
    graduate = (df['Status']=='Graduate').sum()
    dropout_rate = dropout/total*100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Siswa", f"{total:,}")
    col2.metric("Dropout", f"{dropout:,}", f"{dropout_rate:.1f}%")
    col3.metric("Enrolled", f"{enrolled:,}")
    col4.metric("Graduate", f"{graduate:,}", f"{graduate/total*100:.1f}%")

    st.markdown("---")

    # Filter by course
    course_options = ['Semua'] + sorted(df['Course'].unique().tolist())
    selected_course = st.selectbox("Filter berdasarkan Course:", course_options)
    if selected_course != 'Semua':
        df_filtered = df[df['Course']==selected_course]
    else:
        df_filtered = df

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Status Siswa")
        fig, ax = plt.subplots(figsize=(5,3))
        df_filtered['Status'].value_counts().plot(kind='bar', color=['#e74c3c','#f39c12','#2ecc71'], ax=ax)
        ax.set_ylabel('Jumlah Siswa')
        ax.set_title('Distribusi Status')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Dropout Rate per Jurusan")
        course_stats = df.groupby('Course')['Status'].apply(lambda x: (x=='Dropout').mean()*100).sort_values()
        fig, ax = plt.subplots(figsize=(5,3))
        course_stats.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Dropout Rate (%)')
        ax.set_ylabel('Course')
        st.pyplot(fig)
        plt.close()

    st.subheader("Hubungan Faktor dengan Status Siswa")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Distribusi Usia Pendaftaran")
        fig, ax = plt.subplots(figsize=(5,3))
        for status, color in zip(['Graduate','Dropout','Enrolled'], ['#2ecc71','#e74c3c','#3498db']):
            subset = df[df['Status']==status]
            subset['Age_at_enrollment'].hist(bins=15, alpha=0.5, label=status, ax=ax, color=color)
        ax.set_xlabel('Usia Pendaftaran')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.caption("Dropout Rate berdasarkan Beasiswa")
        fig, ax = plt.subplots(figsize=(5,3))
        df.groupby('Scholarship_holder')['Status'].apply(lambda x: (x=='Dropout').mean()*100).plot(kind='bar', color=['coral','seagreen'], ax=ax)
        ax.set_xlabel('Pemegang Beasiswa (0=Tidak, 1=Ya)')
        ax.set_ylabel('Dropout Rate (%)')
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Dropout Rate berdasarkan Tunggakan Biaya (Debtor)")
        fig, ax = plt.subplots(figsize=(5,3))
        df.groupby('Debtor')['Status'].apply(lambda x: (x=='Dropout').mean()*100).plot(kind='bar', color=['coral','seagreen'], ax=ax)
        ax.set_xlabel('Debtor (0=Tidak, 1=Ya)')
        ax.set_ylabel('Dropout Rate (%)')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.caption("Nilai Semester 1 vs Status")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(data=df, x='Status', y='Curricular_units_1st_sem_grade', ax=ax)
        ax.set_title('Nilai Semester 1 vs Status')
        st.pyplot(fig)
        plt.close()

    st.subheader("Nilai Semester 2 vs Status")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(data=df, x='Status', y='Curricular_units_2nd_sem_grade', ax=ax)
    ax.set_title('Nilai Semester 2 vs Status')
    st.pyplot(fig)
    plt.close()

# Predict page
elif page == "Predict":
    st.title("Prediksi Risiko Dropout Siswa")
    st.markdown("Masukkan data siswa untuk memprediksi kemungkinan dropout dan dapatkan rekomendasi intervensi.")

    if model is None:
        st.error("Model belum tersedia. Pastikan file model/model.pkl ada.")
    else:
        with st.form("prediction_form"):
            st.subheader("Data Siswa")

            col1, col2, col3 = st.columns(3)
            with col1:
                marital_status = st.selectbox("Marital Status", [1,2,3,4,5,6], format_func=lambda x: {1:'Single',2:'Married',3:'Widow',4:'Divorced',5:'Union',6:'Separated'}.get(x,str(x)))
                application_mode = st.number_input("Application Mode", min_value=1, max_value=18, value=1)
                application_order = st.number_input("Application Order", min_value=0, value=1)
                course = st.number_input("Course Code", min_value=1, value=171)
                attendance = st.selectbox("Daytime/Evening", [1,0], format_func=lambda x: 'Daytime' if x==1 else 'Evening')
                prev_qual = st.number_input("Previous Qualification", min_value=1, value=1)
                prev_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=120.0)
                nationality = st.number_input("Nacionality", min_value=1, value=1)

            with col2:
                mother_qual = st.number_input("Mother Qualification", min_value=1, value=19)
                father_qual = st.number_input("Father Qualification", min_value=1, value=12)
                mother_occ = st.number_input("Mother Occupation", min_value=1, value=5)
                father_occ = st.number_input("Father Occupation", min_value=1, value=9)
                admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=127.0)
                displaced = st.selectbox("Displaced", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
                special_needs = st.selectbox("Special Needs", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
                debtor = st.selectbox("Debtor", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')

            with col3:
                tuition_up_to_date = st.selectbox("Tuition Fees Up to Date", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
                gender = st.selectbox("Gender", [1,0], format_func=lambda x: 'Male' if x==1 else 'Female')
                scholarship = st.selectbox("Scholarship Holder", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
                age = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=20)
                international = st.selectbox("International", [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
                unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=30.0, value=10.8)
                inflation = st.number_input("Inflation Rate (%)", min_value=-5.0, max_value=10.0, value=1.4)
                gdp = st.number_input("GDP", min_value=-5.0, max_value=10.0, value=1.74)

            st.markdown("#### Nilai Semester 1")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                sem1_credited = st.number_input("Credited", min_value=0, value=0)
                sem1_enrolled = st.number_input("Enrolled", min_value=0, value=0)
            with col_b:
                sem1_evaluations = st.number_input("Evaluations", min_value=0, value=0)
                sem1_approved = st.number_input("Approved", min_value=0, value=0)
            with col_c:
                sem1_grade = st.number_input("Grade", min_value=0.0, max_value=20.0, value=0.0)
                sem1_without = st.number_input("Without Evaluations", min_value=0, value=0)

            st.markdown("#### Nilai Semester 2")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                sem2_credited = st.number_input("Credited", min_value=0, value=0, key='sem2_cred')
                sem2_enrolled = st.number_input("Enrolled", min_value=0, value=0, key='sem2_enr')
            with col_b:
                sem2_evaluations = st.number_input("Evaluations", min_value=0, value=0, key='sem2_eval')
                sem2_approved = st.number_input("Approved", min_value=0, value=0, key='sem2_appr')
            with col_c:
                sem2_grade = st.number_input("Grade", min_value=0.0, max_value=20.0, value=0.0, key='sem2_grd')
                sem2_without = st.number_input("Without Evaluations", min_value=0, value=0, key='sem2_without')

            submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

        if submitted:
            input_data = pd.DataFrame([{
                'Marital_status': marital_status,
                'Application_mode': application_mode,
                'Application_order': application_order,
                'Course': course,
                'Daytime_evening_attendance': attendance,
                'Previous_qualification': prev_qual,
                'Previous_qualification_grade': prev_grade,
                'Nacionality': nationality,
                'Mothers_qualification': mother_qual,
                'Fathers_qualification': father_qual,
                'Mothers_occupation': mother_occ,
                'Fathers_occupation': father_occ,
                'Admission_grade': admission_grade,
                'Displaced': displaced,
                'Educational_special_needs': special_needs,
                'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition_up_to_date,
                'Gender': gender,
                'Scholarship_holder': scholarship,
                'Age_at_enrollment': age,
                'International': international,
                'Curricular_units_1st_sem_credited': sem1_credited,
                'Curricular_units_1st_sem_enrolled': sem1_enrolled,
                'Curricular_units_1st_sem_evaluations': sem1_evaluations,
                'Curricular_units_1st_sem_approved': sem1_approved,
                'Curricular_units_1st_sem_grade': sem1_grade,
                'Curricular_units_1st_sem_without_evaluations': sem1_without,
                'Curricular_units_2nd_sem_credited': sem2_credited,
                'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
                'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
                'Curricular_units_2nd_sem_approved': sem2_approved,
                'Curricular_units_2nd_sem_grade': sem2_grade,
                'Curricular_units_2nd_sem_without_evaluations': sem2_without,
                'Unemployment_rate': unemployment,
                'Inflation_rate': inflation,
                'GDP': gdp
            }])

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0, 1]
            status = 'Dropout' if pred == 1 else 'Graduate'

            st.markdown("---")
            st.subheader("Hasil Prediksi")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediksi Status", status)
            col2.metric("Probabilitas Dropout", f"{prob:.1%}")
            col3.metric("Probabilitas Graduate", f"{1-prob:.1%}")

            # Recommendation
            st.markdown("### Rekomendasi Intervensi")
            if prob > 0.7:
                st.error("Risiko dropout **SANGAT TINGGI**. Segera lakukan intervensi: bimbingan akademik intensif, konseling psikologis, dan evaluasi bantuan finansial.")
            elif prob > 0.5:
                st.warning("Risiko dropout **CUKUP TINGGI**. Pertimbangkan bimbingan tambahan, pengingat jadwal, dan pemantauan performa akademik lebih closely.")
            elif prob > 0.3:
                st.info("Risiko dropout **SEDANG**. Terus pantau performa dan berikan dukungan preventif.")
            else:
                st.success("Risiko dropout **RENDAH**. Siswa menunjukkan profil yang baik. Pertahankan dukungan yang ada.")

            # Feature contributions
            st.markdown("### Faktor yang Mempengaruhi Prediksi")
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importances = model.named_steps['classifier'].feature_importances_
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7,4))
            sns.barplot(data=fi_df, y='feature', x='importance', palette='viridis', ax=ax)
            ax.set_title('Top 10 Faktor Paling Berpengaruh')
            st.pyplot(fig)
            plt.close()
