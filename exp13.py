
import os
import base64

from PIL import Image
import pytesseract

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# Step 1: Configure API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured.")


# Step 2: Configure Tesseract OCR
# Change this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# Step 3: Load Groq Vision Model
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=api_key,
    temperature=0
)


# Step 4: Load image
IMAGE_PATH = "blood_work.png"

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


# Step 5: Send image to Vision Model
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                "This is a blood work report. "
                "Extract all test results and identify "
                "values outside the reference ranges shown "
                "in the report. Do not diagnose."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        }
    ]
)

response = llm.invoke([message])

print("\n===== VISION MODEL OUTPUT =====")
print(response.content)


# Step 6: Extract text using OCR
image = Image.open(IMAGE_PATH)

extracted_text = pytesseract.image_to_string(image)

print("\n===== OCR EXTRACTED TEXT =====")
print(extracted_text)


# Step 7: Send extracted text to LLM
ocr_message = HumanMessage(
    content=(
        "This is text extracted from a blood work report.\n"
        "Extract all test results and identify values "
        "outside the reference ranges provided.\n"
        "Do not diagnose.\n\n"
        "Report text:\n"
        + extracted_text
    )
)

ocr_response = llm.invoke([ocr_message])

print("\n===== OCR + LLM OUTPUT =====")
print(ocr_response.content)
