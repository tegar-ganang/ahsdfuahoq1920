import fitz  # PyMuPDF
import json
import os
import pandas as pd
import traceback
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Get API Key
KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=KEY)

# Define Templates
TEMPLATE = """
Text:{text}
Anda adalah seorang ahli pembuat MCQ (multiple choice question). Dengan teks di atas, tugas Anda adalah membuat kuis yang terdiri dari {number} pertanyaan pilihan ganda untuk siswa {subject} dengan nada {tone}. 
Pastikan pertanyaan-pertanyaan tersebut tidak diulang dan periksa semua pertanyaan agar sesuai dengan teks.
Pastikan untuk memformat jawaban Anda seperti RESPONSE_JSON di bawah ini dan gunakan sebagai panduan. 
Pastikan untuk membuat {number} MCQ
### RESPONSE_JSON
{response_json}
"""

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    }
}

TEMPLATE2 = """
Anda adalah seorang ahli tata bahasa dan penulis bahasa Indonesia. Diberikan Kuis Pilihan Ganda untuk siswa {subject}. 
Anda perlu mengevaluasi kompleksitas pertanyaan dan memberikan analisis lengkap dari kuis tersebut. Hanya gunakan maksimal 50 kata untuk analisis kompleksitas. 
Jika kuis tidak sesuai dengan kemampuan kognitif dan analisis siswa, 
perbarui pertanyaan kuis yang perlu diubah dan ubah nadanya sehingga sesuai dengan kemampuan siswa.

Quiz_MCQs:
{quiz}

Simak jawaban dari seorang penulis Indonesia yang ahli dalam kuis di atas:
"""

def call_gemini_api(prompt):
    """Function to call Gemini API and get a response."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def clean_json_response(response):
    """
    Cleans AI-generated JSON response by removing unwanted prefixes, suffixes, and extra data.
    """
    response = response.strip()

    # Remove Markdown JSON formatting like ```json ... ```
    if response.startswith("```json"):
        response = response[7:]  # Remove ```json
    elif response.startswith("```"):
        response = response[3:]  # Remove ```

    # Find the first valid JSON object
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)  # Extract only the JSON portion

    # Remove trailing backticks or any extra text
    response = re.sub(r"```$", "", response.strip())

    return response

def parse_quiz_json(quiz_response):
    """
    Parses and loads the cleaned JSON response.
    """
    try:
        cleaned_response = clean_json_response(quiz_response)
        quiz_json = json.loads(cleaned_response)
        return quiz_json
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Raw Response:", quiz_response)  # Debugging log
        return None
    
def generate_and_evaluate_mcqs(uploaded_file, number, subject, tone):
    try:
        pdf_text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()

        # Step 1: Generate MCQs using Gemini API
        formatted_prompt = TEMPLATE.format(
            text=pdf_text,
            number=number,
            subject=subject,
            tone=tone,
            response_json=json.dumps(RESPONSE_JSON)
        )

        # Call Gemini API to generate MCQs
        quiz_response = call_gemini_api(formatted_prompt)
        if not quiz_response:
            raise ValueError("Failed to get response from Gemini API")

        # Debugging log to check the content of quiz
        print("Quiz JSON content before cleaning:", quiz_response)

        # Extract and clean JSON response
        quiz_response = clean_json_response(quiz_response)
        quiz_json = parse_quiz_json(quiz_response)

        # Debugging log to check the content of quiz after cleaning
        print("Quiz JSON content after cleaning:", quiz_json)

        # Ensure quiz_json is valid before proceeding
        if not isinstance(quiz_json, dict):
            raise ValueError("Quiz JSON is not a valid dictionary")

        quiz_table_data = []
        for key, value in quiz_json.items():
            mcq = value["mcq"]
            options = " | ".join([f"{option}: {option_value}" for option, option_value in value["options"].items()])
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        # Generate evaluation prompt
        evaluation_prompt = TEMPLATE2.format(subject=subject, quiz=json.dumps(quiz_json, indent=4))
        
        # Call Gemini API to evaluate MCQs
        evaluation_response = call_gemini_api(evaluation_prompt)
        if not evaluation_response:
            raise ValueError("Failed to get evaluation response from Gemini API")

        return {
            "quiz": quiz_table_data,
            "evaluation": evaluation_response
        }

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(traceback.format_exc())
        return {"error": "Invalid JSON response from Gemini API"}
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}