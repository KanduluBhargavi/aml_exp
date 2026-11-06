
import os
from langchain_groq import ChatGroq

# Step 1: Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Please set your GROQ_API_KEY")

# Step 2: Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)

# Step 3: Read blood test report
with open("blood_work.txt", "r", encoding="utf-8") as f:
    blood_report = f.read()

print("Blood Test Report:")
print(blood_report)

# Step 4: Create extraction prompt
extraction_prompt = f"""
You are a medical report extraction assistant.

From the blood report below, extract all test values
and classify each one as High, Low, or Normal.

Use the reference ranges provided in the report.

Format your response as:

Test Name: Value
Status: High/Low/Normal
Reference Range: Range

Blood Report:
{blood_report}
"""

# Step 5: Extract test results
extraction_response = llm.invoke(extraction_prompt)

print("\n===== BLOOD REPORT ANALYSIS =====")
print(extraction_response.content)

# Step 6: Create diet recommendation prompt
diet_prompt = f"""
You are a clinical nutrition assistant specializing
in Indian dietary habits.

Based on the blood test analysis below, provide:

1. A short health summary in 4-5 lines
   explaining the patient's results in simple language.

2. A practical diet plan with only these two sections:

   a) Foods to avoid
   b) Foods to eat more of

Do not add any other sections.

Use simple language and avoid making a medical diagnosis.
Mention that dietary recommendations should be reviewed
by a qualified healthcare professional.

Blood Test Analysis:
{extraction_response.content}
"""

# Step 7: Generate diet recommendation
diet_response = llm.invoke(diet_prompt)

print("\n===== DIET RECOMMENDATION =====")
print(diet_response.content)
