import pandas as pd
import streamlit as st
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_column', None)


custom_css = """
<style>
.css-1l02z2j {
    display: block !important;
    width: 20% !important;
}
</style>
"""

# Display the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


st.write("""
<style>
  .centered-text {
    text-align: center;
  }
</style>

# <div class="centered-text">INX Future Inc Employee Performance</div>
""", unsafe_allow_html=True)


# Load the data
data = pd.read_csv('data.csv')
data.drop(columns= ['EmpNumber', 'PerformanceRating'], inplace= True)

st.sidebar.write("""
# Select Features
""")

## User Input
imputed_data = {}

for col in data.columns:
    if data[col].dtype == 'O': 
        imputed_data[col] = st.sidebar.selectbox(
            f"Select {col}", data[col].unique())
        
    elif (data[col].dtype in ['int64', 'float64']) and (data[col].nunique() < 5):  
            imputed_data[col] = st.sidebar.selectbox(
                f"Select {col}", data[col].unique())

    else:
        if data[col].dtype in ['int64', 'float64']:  
            min_val = data[col].min()
            max_val = data[col].max()
            imputed_data[col] = st.sidebar.slider(
                f"Select {col}", min_val, max_val)


                
user_input = pd.DataFrame(imputed_data, index=[0])
st.header('User Input Data')
st.dataframe(user_input)



# User Input Processing

# Original encoding dictionary
encoding_dict = {'Other': 0, 'Life Sciences': 1, 'Marketing': 2, 'Human Resources': 3, 'Technical Degree': 4, 'Medical': 5}
user_input['EducationBackground'] = encoding_dict.get(user_input['EducationBackground'].values[0])


label_encoder = joblib.load('encoder_model.joblib')
for col, model_info in label_encoder.items():
    encoder_model = model_info['model']
    user_input[col] = encoder_model.transform(user_input[col])


# Load the power transformer
power_trans = joblib.load('power_transform.joblib')

# Specify columns to transform
col_trans = ['Age', 'EmpEducationLevel', 'EducationBackground', 'EmpEnvironmentSatisfaction', 'EmpHourlyRate',
             'EmpJobInvolvement', 'EmpJobSatisfaction', 'EmpRelationshipSatisfaction', 'TrainingTimesLastYear',
             'EmpWorkLifeBalance']

# Extract the columns to transform
col_to_trans = user_input[col_trans]

# Transform the selected columns
user_input[col_trans] = power_trans.transform(col_to_trans) 



# Standardize the features
scaler = joblib.load('scaled.joblib')
user_input_scaled = scaler.transform(user_input)



# Save the PCA model
pca = joblib.load('pca_model.joblib')
user_input_pca = pca.transform(user_input_scaled)

st.header("Transformed User Input")
st.dataframe(user_input_pca)


# Load the trained model
with open('random_for_class_hyp.pickle', 'rb') as file:
    rand_for_best_para = pickle.load(file)



output = rand_for_best_para.predict(user_input_pca)
st.header('Final Prediction')
st.dataframe(output)

output_proba = rand_for_best_para.predict_proba(user_input_pca)
st.header('Final Prediction')
st.dataframe(output_proba)