import streamlit as st
import pandas as pd
import os
import core_functions as cf

# Function to check if the mega CSV file exists
def mega_csv_exists(mega_csv_path):
    return os.path.exists(mega_csv_path)

# Title of the app
st.title("AutoGradeSync")

# Section 1: Sync Academic Records
st.header("1. Sync Academic Records")

# Step 1: Enter number of subjects
num_subjects = st.number_input("Enter the number of subjects:", min_value=1, step=1)

# Step 2: Input subject names
subject_names = []
for i in range(num_subjects):
    subject = st.text_input(f"Enter the name of subject {i + 1}:", key=f"subject_name_{i}")
    subject_names.append(subject)

# Step 3: Upload PDFs
pdf_files = []
for i, subject in enumerate(subject_names):
    uploaded_file = st.file_uploader(f"Upload PDF for {subject}", type=["pdf"], key=f"file_uploader_{i}")
    
    if uploaded_file is not None:
        pdf_files.append(uploaded_file)

# Step 4: Sync Data button
if st.button("Sync Data"):
    if len(pdf_files) == num_subjects:  # Ensure all PDFs are uploaded
        # Call your sync_data function here with pdf_files and subject_names
        cf.sync_data(pdf_files, subject_names) # Uncomment when you have the function
        st.success("Data synced successfully!")
    else:
        st.error("Please upload PDFs for all subjects before syncing.")
        
        
# Section 2: 
st.header("2. Student Grade Lookup")

if mega_csv_exists('mega_grades.csv'):
    student_id = st.text_input("Enter Student ID:")
    
    if st.button("Lookup Grades"):
        # Call search_student_data function with student_id
        student_data = cf.search_student_data(student_id, 'mega_grades.csv')
        
        if isinstance(student_data, str):  # If the return type is a string, it means not found
            st.error(student_data)
        else:
            st.write(student_data)
            # Generate and display the barplot for the student
            cf.plot_student_barplot(student_data)
else:
    st.warning("Please sync academic records first!")
    
# Section 3: Grade Distributions
st.header("3. Grade Distributions")

if mega_csv_exists('mega_grades.csv'):
    if st.button("Generate Grade Distributions"):
        # Call your function to generate histograms here
        cf.plot_subject_histograms('mega_grades.csv')
        st.success("Distributions generated!")
else:
    st.warning("Please sync academic records first!")