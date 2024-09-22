import pandas as pd
from openai import OpenAI
import tabula
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os  # To check if the file exists

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Assuming pdf_files contains the list of PDFs and subject_names contains respective subjects
# pdf_files = ['Calculus_Coursework Final.pdf', 'OOPDS_Coursework Final.pdf', 'OOAD_Coursework Final.pdf']
# subject_names = ['Calculus','OOPDS', 'OOAD']
mega_csv_path = 'mega_grades.csv'  # Path to the mega CSV file

# Function to save the final dataframe to the mega CSV
def save_to_mega_csv(final_df, mega_csv_path):
    # Check if the mega CSV file already exists
    if os.path.exists(mega_csv_path):
        # Read the existing CSV into a DataFrame
        existing_df = pd.read_csv(mega_csv_path)
        
        # Check for duplicate rows between the final_df and existing_df
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
        
        # Drop duplicate rows (keeping only unique entries)
        combined_df.drop_duplicates(subset=['Student ID', 'Subject'], inplace=True)

        # Write back to the CSV (overwrite it with the combined data)
        combined_df.to_csv(mega_csv_path, mode='w', index=False)
        print("File exists, it is appended:\n")
        print(combined_df)
    else:
        # If the file doesn't exist, create it with headers
        final_df.to_csv(mega_csv_path, mode='w', index=False)
        print("File does not exist, file created is:\n")
        print(final_df)
      
        
# Function to extract tables from PDF using Tabula
def extract_tables(pdf_file_path, subject_name):
    tables = tabula.read_pdf(pdf_file_path, pages='all') # Read all pages of the PDF, returns a list of dataframes (tables)
    column_names = tables[0].columns # Get the column names of the first table
    for i in range(1, len(tables)): # Loop through the remaining tables
        tables[i].columns = column_names # Set the column names of each table to match the first table
    combined_df = pd.concat(tables, ignore_index=True) # Concatenate all tables into a single dataframe
    return combined_df


#Function to extract relevant data using GPT-4
def gpt_extract_relevant_data(raw_table_data, subject_name):
    # Define your GPT prompt here
    prompt = f"""
    The following is a raw table of student data extracted from a PDF. 
    Please extract the 'Student ID' and 'Total Marks' for the subject '{subject_name}'.
    Here is the raw table:
    {raw_table_data}
    Please return the data in the following format: 'Student ID | Total Marks'.
    For the total marks, specify it as based on marks obtained out of the total marks available.
    For example, if heading is 'Marks (100)', and student scored 75 marks, it should be represented as '75/100'.
    Do not give any other information. simply return the data in the specified format
    Do not leave any space between rows.
    """
    # Assuming you have the OpenAI API integration ready, get GPT-4 response
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': 'You are an established statistician helping a teacher extract student data from a PDF.'},
            {'role': 'user', 'content': f'{prompt}'},
        ],
    )
    # Extract the returned data from the GPT response
    extracted_data = response.choices[0].message.content
    print(response.usage)
    return extracted_data


# Function to create a new DataFrame and append GPT-processed data
def append_to_final_dataframe(extracted_data, subject_name, final_df):
    # List to hold the new rows of data
    data_rows = []
    
    lines = extracted_data.splitlines()
    
    # Process each row in the extracted data
    for row in lines:
        student_id, total_marks = row.split('|')
        student_id = student_id.strip()
        total_marks = total_marks.strip()
        
        # Store each row in a dictionary with corresponding subject
        data_rows.append({
            'Student ID': student_id,
            'Total Marks': total_marks,
            'Subject': subject_name
        })
    
    # Convert the list of dictionaries to a DataFrame
    new_df = pd.DataFrame(data_rows)
    
    # Concatenate the new data to the final DataFrame
    final_df = pd.concat([final_df, new_df], ignore_index=True)
    return final_df


# Function to synch data into the mega CSV file
def sync_data(pdf_files, subject_names):
    # Initialize an empty DataFrame to store final data
    final_df = pd.DataFrame(columns=['Student ID', 'Total Marks', 'Subject']) # Columns for the final DataFrame
    
    for pdf_file, subject in zip(pdf_files, subject_names):
        # Step 1: Extract raw data using Tabula
        raw_table_data = extract_tables(pdf_file, subject).to_csv(sep='|', index=False) # Convert the DataFrame to a CSV string for analysis
        
        # Step 2: Use GPT-4 to extract relevant data
        extracted_data = gpt_extract_relevant_data(raw_table_data, subject)

        # Step 3: Append the extracted data to the final DataFrame
        final_df = append_to_final_dataframe(extracted_data, subject, final_df)

        # Step 4: Save the final DataFrame to the mega CSV file
        save_to_mega_csv(final_df, mega_csv_path)
        # After the loop, the mega CSV will contain all processed data
        
        
# Function to search for a student's data in the mega CSV
def search_student_data(student_id, mega_csv_path):
    # Load the mega CSV file into a DataFrame
    df = pd.read_csv(mega_csv_path)
    
    # Search for the student ID and return the rows
    student_data = df[df['Student ID'] == student_id]
    
    if student_data.empty:
        return f"Student ID: {student_id} not found."
    else:
        return student_data # Return the DataFrame containing the student's data, convert to string when printing

    
# Function to plot a bar plot of a student's performance by subject
def plot_student_barplot(student_data):
    # Extract subjects and calculate relative marks as percentages
    subjects = student_data['Subject'] #student_data MUST BE a dataframe
    
    # Safely handle different formats of Total Marks
    def calculate_percentage(mark):
        if isinstance(mark, str) and '/' in mark:  # Check if the mark is in "marks/total_marks" format
            obtained, total = map(float, mark.split('/'))
            return (obtained / total) * 100
        return mark  # Return the mark as is if it's already a percentage
    
    marks = student_data['Total Marks'].apply(calculate_percentage)

    # Create a DataFrame for Seaborn
    data = pd.DataFrame({'Subject': subjects, 'Marks (%)': marks})

    # Adjust the figure size (width, height in inches)
    plt.figure(figsize=(3, 4))  # width, height
    ax = sns.barplot(x='Subject', y='Marks (%)', data=data, palette= "crest")
    
    ax.set_ylim(0, 100)
    
    # Use Matplotlib's bar_label to annotate bars with their values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=7)


    # Customize the plot
    plt.xlabel('Subjects', fontsize=8)
    plt.ylabel('Marks Obtained (%)', fontsize=8)
    plt.title('Student Performance by Subject', fontsize=8)
    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(fontsize=8)  
    plt.tight_layout()

    # Display the plot in Streamlit
    with st.expander("View Student relative performance by subject"):
        st.pyplot(plt)


#Function to plot histograms of marks distribution for each subject
def plot_subject_histograms(mega_csv_path):
    # Load the mega CSV file into a DataFrame
    df = pd.read_csv(mega_csv_path)
    
    # Convert the 'Total Marks' column into a format for analysis (e.g., 30/40 -> percentage)
    df['Percentage Marks'] = df['Total Marks'].apply(lambda x: float(x.split('/')[0]) / float(x.split('/')[1]) * 100)

    # Get unique subjects
    subjects = df['Subject'].unique()

    # Loop through each subject and plot a histogram
    for subject in subjects:
        subject_data = df[df['Subject'] == subject]['Percentage Marks']

        plt.figure(figsize=(8, 6))
        sns.histplot(subject_data, kde=False, bins=10, color='teal')
        plt.title(f'Marks Distribution for {subject}')
        plt.xlabel('Percentage Marks')
        plt.ylabel('Number of Students')
        plt.xlim(0, 100)
        plt.show()
        
        with st.expander(f"View marks distribution for {subject}"):
            st.pyplot(plt)
            plt.clf()  # Clear the figure to avoid overlay