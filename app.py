import sys
import os
import streamlit as st
import pandas as pd
from slaq_extractor import generate_and_evaluate_questions
from io import BytesIO
import traceback

sys.path.append("C:/Users/User-AP/Downloads/questiongenerator")

st.title("Question Generator from PDF")
st.write("Upload a PDF file to extract Short and Long Answer Questions and generate a CSV file.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
number = st.number_input("Number of Questions", min_value=1, step=1)
subject = st.text_input("Subject")
tone = st.text_input("Tone")
question_type = st.selectbox("Question Type", ["Short Answer", "Long Answer"])
grade_level = st.selectbox("Grade Level", ["SD", "SMP", "SMA"])

if st.button("Generate Questions"):
    if uploaded_file and number and subject and tone and question_type and grade_level:
        try:
            questions = generate_and_evaluate_questions(uploaded_file, number, subject, tone, question_type, grade_level)
            if questions:
                df = pd.DataFrame(questions["quiz"])
                st.write("### Generated Questions")
                st.dataframe(df)  # Display Questions in a table
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=BytesIO(csv.encode('utf-8')),
                    file_name='questions.csv',
                    mime='text/csv'
                )
                st.write("### Evaluation Feedback")
                st.write(questions["evaluation"])
            else:
                st.error("No questions extracted from the PDF.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())
    else:
        st.error("Please upload a PDF file and fill all the fields.")
