import fitz  # PyMuPDF
import json
import os
import traceback
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Get API Key
KEY = os.getenv("GEMINI_API_KEY")

# Initialize LangChain's Google Generative AI model
llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=KEY, temperature=0.5)

# Define Prompt Templates
TEMPLATE_GEN = """
Text:{text}
Anda adalah seorang ahli pembuat soal. Dengan teks di atas, tugas Anda adalah membuat kuis yang terdiri dari {number} pertanyaan {question_type} untuk siswa {subject} tingkat {grade_level} dengan nada {tone}. 
Pastikan pertanyaan-pertanyaan tersebut tidak diulang dan periksa semua pertanyaan agar sesuai dengan teks.
Pastikan untuk memformat jawaban Anda seperti RESPONSE_JSON di bawah ini dan gunakan sebagai panduan. 
Pastikan kuis menerapkan HOTS (Higher Order Thinking Skills).
Pastikan untuk membuat {number} pertanyaan.

### RESPONSE_JSON
{response_json}
"""

TEMPLATE_EVAL = """
Anda adalah seorang ahli tata bahasa dan penulis bahasa Indonesia. Diberikan kumpulan pertanyaan {question_type} untuk siswa {subject} tingkat {grade_level}. 
Anda perlu mengevaluasi kompleksitas pertanyaan dan memberikan analisis lengkap dari kuis tersebut. Hanya gunakan maksimal 50 kata untuk analisis kompleksitas. 
Jika kuis tidak sesuai dengan kemampuan kognitif dan analisis siswa, 
perbarui pertanyaan kuis yang perlu diubah dan ubah nadanya sehingga sesuai dengan kemampuan siswa. Terapkan HOTS (Higher Order Thinking Skills).


Quiz:
{quiz}

Simak jawaban dari seorang penulis Indonesia yang ahli dalam kuis di atas:
"""

RESPONSE_JSON = {
    "1": {"question": "Pertanyaan pendek di sini", "expected_answer": "Jawaban yang benar. Tidak lebih dari 5 kata"},
    "2": {"question": "Pertanyaan panjang di sini", "expected_answer": "Jawaban yang lebih kompleks. Minimal 1 paragraf"}
}

# Define Prompt Templates in LangChain
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "question_type", "grade_level", "response_json"],
    template=TEMPLATE_GEN
)

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz", "question_type", "grade_level"],
    template=TEMPLATE_EVAL
)

# Define LangChain Processing
quiz_chain = quiz_generation_prompt | llm | StrOutputParser(key="quiz")
review_chain = quiz_evaluation_prompt | llm | StrOutputParser(key="review")

# Define Parallel Execution of Question Generation and Evaluation
generate_evaluate_chain = RunnableParallel(quiz=quiz_chain, review=review_chain)

def clean_json_response(response):
    """Cleans AI-generated JSON response by removing unwanted prefixes, suffixes, and extra data."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        response = match.group(0)
    response = re.sub(r"```$", "", response.strip())
    return response

def parse_quiz_json(quiz_response):
    """Parses and loads the cleaned JSON response."""
    try:
        cleaned_response = clean_json_response(quiz_response)
        quiz_json = json.loads(cleaned_response)
        return quiz_json
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Raw Response:", quiz_response)
        return None

def generate_and_evaluate_questions(uploaded_file, number, subject, tone, question_type, grade_level):
    try:
        pdf_text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()

        # Step 1: Generate Questions
        quiz_response = quiz_chain.invoke({
            "text": pdf_text,
            "number": number,
            "subject": subject,
            "tone": tone,
            "question_type": question_type,
            "grade_level": grade_level,
            "response_json": json.dumps(RESPONSE_JSON)
        })

        if not quiz_response:
            raise ValueError("Question generation failed")

        print("Quiz JSON content before cleaning:", quiz_response)
        quiz = clean_json_response(quiz_response)
        quiz_json = parse_quiz_json(quiz)

        if not isinstance(quiz_json, dict):
            raise ValueError("Quiz JSON is not a valid dictionary")

        # Step 2: Evaluate Questions
        evaluation_response = review_chain.invoke({
            "subject": subject,
            "quiz": json.dumps(quiz_json, indent=4),
            "question_type": question_type,
            "grade_level": grade_level
        })

        if not evaluation_response:
            raise ValueError("Question evaluation failed")

        print("Evaluation Response:", evaluation_response)

        quiz_table_data = []
        for key, value in quiz_json.items():
            question = value["question"]
            expected_answer = value["expected_answer"]
            quiz_table_data.append({"Question": question, "Expected Answer": expected_answer})

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
