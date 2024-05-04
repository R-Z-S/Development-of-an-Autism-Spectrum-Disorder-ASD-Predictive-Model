Development of an Autism Spectrum Disorder (ASD) Predictive Model and Diagnostic Web Application

Introduction

This project focuses on developing a machine learning model and a user-friendly web application for predicting Autism Spectrum Disorder (ASD) and facilitating diagnostic consultations. The primary objective is to create an accessible tool for individuals seeking ASD assessment and professional guidance.

Project Structure
Dataset

The dataset used for training the predictive model contains information related to ASD diagnosis, including demographic details, medical history, developmental milestones, and behavioral assessments.

Machine Learning Model

Various machine learning algorithms, such as Support Vector Machine (SVM), Logistic Regression, Random Forest, Decision Tree, and K-Nearest Neighbors (KNN), were employed to develop the predictive model.
The model was trained on the dataset to predict the likelihood of ASD based on input features.

Web Application

A user-friendly web application was developed to integrate the trained machine learning model for real-time ASD prediction.
The application offers features such as registration, login, risk assessment form, dashboard for visualization, and contact page for consultations.

Requirements

Install libraries
#pip install streamlit
#pip install pandas
#pip install sklearn
pandas==1.1.4
matplotlib==3.3.3
seaborn==0.11.1
numpy==1.18.5
streamlit==0.72.0
plotly==4.14.1
Pillow==8.0.1
scikit_learn==0.24.0

Getting Started

Clone the repository to your local machine.
Install the necessary Python libraries listed in the requirements.
Run the web application using Streamlit by executing the command - streamlit run Register.py
Sign up or log in to access the features of the web application.
Input relevant information for ASD prediction in the risk assessment form.
Explore the dashboard for visualizations of ASD traits and statistics.
Contact a healthcare professional for consultation and appointments using the provided contact page.

Folder Structure

Register.py: Main script for running the web application.
data.db: Database file storing login information.
asd_data_csv.csv: Dataset containing ASD-related information.
Predict_Autism_Spectrum_Disorder_ML.ipynb: Jupyter Notebook file for machine learning prediction.
pages/: Directory containing Python scripts for web pages.
Home.py: Script for the home page.
Dashboard.py: Script for the dashboard page.
Contact.py: Script for the contact page.
Form.py: Script for the risk assessment form page.
images/: Directory containing images related to ASD.
style.css: CSS stylesheet for styling HTML pages.
 
Additional Notes

Feel free to modify the code to experiment with different machine learning algorithms or feature engineering techniques.
Contributions, feedback, and suggestions are welcome through pull requests or issue reports.
Ensure adherence to ethical guidelines regarding data privacy and confidentiality while handling sensitive information.

Conclusion

The ASD predictive model and diagnostic web application offer a convenient and reliable tool for assessing ASD likelihood and facilitating consultations with healthcare professionals. By leveraging machine learning techniques and user-friendly interface design, the project aims to promote proactive healthcare management and support individuals and families affected by ASD.
 

![image](https://github.com/R-Z-S/Development-of-an-Autism-Spectrum-Disorder-ASD-Predictive-Model/assets/140642507/bc9eeee5-f81c-43ab-945c-23ff44f97f57)
