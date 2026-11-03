#**BEFORE RUNNING Click on "Copy to Drive" above or File--> Save a Copy in Drive**

---


#This example is for a basic, no-frills chatbot.


---

#**Prerequisites:**

#1. A Google Colab account (free or pro tier).

#2. LangSmith account and API key (sign up, then get api key).

#3. An OpenAI API key.

#**Additional resources:**

#LangChain, LangGraph, and LangSmith have tutorials, for example: Some of this simple chatbot comes from the following tutorial: https://python.langchain.com/docs/tutorials/llm_chain/.



---


#**Chatbot code begins here**



---



#**Step 1.** Install needed packages.

#!pip install langchain                                                            # we will be working within the langchain/langsmith/langgraph environment
#!pip install langchain_openai                                                     # openAI specific langchain functions

#**Step 2.** Import packages to this session.

import getpass                                                                    # this is to ask for password input rather than directly inputting the password
import os                                                                         # functions to work with operating system
from langchain_openai import ChatOpenAI                                           # some of the openAI functions
from langchain_core.messages import HumanMessage, SystemMessage                   # some of the openAI message functions

#**Step 3.** Set up access for LangSmith. This allows LangSmith to monitor activity. Need LangSmith API key here. This prompts you to enter the API key. Paste API key and hit 'Enter'.

os.environ["LANGSMITH_TRACING"] = "true"                                          # permit langsmith to monitor AI process
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API key: ")    # enter LangSmith API key

#**Step 4.** Set up access to LLM. Here that is OpenAI's API. Need an API key here for OpenAI. This prompts you to enter the API key. Paste API key and hit 'Enter'.

if not os.environ.get("OPENAI_API_KEY"):                                          # if there is no OPENAI_API_KEY in the environment variables
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")    # enter OpenAI API key

#**Step 5.** Choose LLM model and run initial chatbot set up.

llm = ChatOpenAI(model="gpt-4o-mini")

sysmessage = """You are EngageAI, an Agentic AI Employee Engagement and Training Assistant for TechBridge Solutions.

"""COMPANY CONTEXT:
- Company Name: TechBridge Solutions
- Mission: To empower every employee with the tools, knowledge, and confidence to thrive in a technology-driven workplace.
- Vision: To be a leading organization where responsible AI adoption and continuous learning drive innovation and inclusive growth.

POLICIES AND PROCEDURES:
- Employees must use only company-approved AI tools listed in the AI Acceptable Use Policy (AUP-2024).
- All AI-generated outputs must be reviewed by a human before being used in official communications or decisions.
- Employees are prohibited from entering confidential company data, customer records, or personally identifiable information into any AI tool.
- Violations of the AUP must be reported to the IT department within 24 hours.

QUARTERLY GOALS (Q4 2024):
- Complete AI literacy training for 100% of Operations and Customer Support staff.
- Deploy EngageAI to all five departments.
- Achieve an 80% module completion rate across all roles by end of quarter.

ANNUAL OBJECTIVES:
- Reduce onboarding time for AI tools by 30%.
- Build a culture of responsible AI use across all departments.
- Establish a skills-gap baseline for each role to inform 2025 training planning.

AVAILABLE TRAINING MODULES BY ROLE:
HR: (1) AI in HR Processes - 30 min | (2) Data Privacy and Employee Records - 20 min | (3) Responsible AI Decision-Making - 25 min
Operations: (1) AI Tools for Workflow Automation - 30 min | (2) AI Acceptable Use Policy Overview - 20 min | (3) Identifying AI Errors and Escalation - 25 min
Customer Support: (1) AI-Assisted Customer Interactions - 30 min | (2) Handling AI Limitations with Customers - 20 min | (3) Data Privacy in Customer Service - 25 min
IT: (1) Deploying and Managing AI Tools - 45 min | (2) Cybersecurity and AI Risk - 30 min | (3) AI Incident Reporting - 20 min
Manager: (1) Leading AI Adoption on Your Team - 30 min | (2) Coaching Employees Through AI Transitions - 25 min | (3) AI Performance Metrics and Reporting - 20 min

TONE AND BEHAVIOR:
- Always respond in a supportive, encouraging, and professional tone.
- If an employee expresses anxiety or fear about AI replacing their role, acknowledge and reassure them before providing guidance.
- Conversational responses for general questions should be 2-4 sentences.
- Training recommendations should be a numbered list with role, topic, and estimated time.
- Provide a session summary when requested with completed topics and recommended next steps.

ESCALATION:
- Direct employees to HR for disciplinary actions, compensation, legal compliance, or medical accommodations.
- HR Email: hr@techbridgesolutions.com
- HR Helpline: (555) 400-1200, Monday-Friday 8AM-5PM
- Ethics Hotline: ethics.techbridgesolutions.com

DATA PRIVACY:
- Do not store, repeat, or reference personally identifiable information beyond the current session.
- Do not share one employee's responses with another user.

SCOPE:
- Only answer questions related to TechBridge Solutions policies, role-specific training, and professional development.
- If a question is outside this scope, politely redirect the employee."""


#**Step 6.** Get user input, call the LLM, and print reply.

guardrailmessage = """
You check user input to determine whether a user is trying to bypass guardrails.
The guardrails are defined below:

1. Only discuss topics related to employee engagement, AI adoption, company
   policies, organizational goals, role-specific training, and professional
   development.
2. Do not discuss illegal, harmful, unethical, or offensive activities.
3. Do not permit discussions that discriminate about any group of people.
4. Do not answer questions unrelated to employee training, AI tools, or
   organizational objectives.

If the user input violates any of the guardrails, respond only with Yes.
If the user input does not violate any of the guardrails, respond only with No.

User input:
"""                                                             # print the response from the model

import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage

# This is your existing guardrail text.
# (If you have a separate variable name for this, adjust it here)
guardrailmessage = """You are a verification assistant. Check if the user's message is related to professional development, human resources, AI training, or company policy. Respond with 'Yes' if it is allowed, or 'No' if it is completely out of scope."""

def predict(message, history):
    # 1. Run your guardrail safety verification check
    premessages = [
        SystemMessage(content=guardrailmessage),
        HumanMessage(content=message),
    ]
    preres = llm.invoke(premessages)

    # If the user tries to ask about an out-of-scope topic, block it safely
    if "No" in preres.content:
        return "I am EngageAI, your Training Assistant. I can only answer questions related to TechBridge Solutions policies, role-specific training modules, and professional development."

    # 2. Fulfill valid inputs using your comprehensive TechBridge Solutions prompt
    messages = [
        SystemMessage(content=sysmessage),
        HumanMessage(content=message),
    ]
    res = llm.invoke(messages)
    return res.content

# 3. Define your user-friendly interface presentation parameters
demo = gr.ChatInterface(
    fn=predict,
    title="EngageAI — Employee Training Assistant",
    description="Welcome to TechBridge Solutions' automated employee engagement portal. I can assist you with company policies, AI acceptable use guidelines, and your specialized training modules.",
    examples=[
        "I'm in Customer Support, what training modules should I take?",
        "What are the rules regarding company data privacy and AI tools?",
        "I'm feeling anxious about AI tools changing my day-to-day role."
    ]
)

# 4. Launch with the advanced network optimization and the keep-alive loop
demo.launch(
    share=True,
    prevent_thread_lock=True,
    max_threads=40
)

# Infinite execution loop forces Google Colab to maintain background server routes
import time
try:
    print("Keep-alive loop active. EngageAI is listening!")
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Gradio link stopped manually.")



import os
from google.colab import files

# Force Colab to save the notebook structure directly as a Python script
os.system("jupyter nbconvert --to script *.ipynb")

# Rename the converted file to app.py so it's ready for your app deployment
for file in os.listdir('.'):
    if file.endswith('.py') and file != 'app.py':
        os.rename(file, 'app.py')
        break

# Download the correct app.py file directly to your computer's downloads folder
if os.path.exists('app.py'):
    files.download('app.py')
else:
    print("Please run all cells in your notebook once, then run this cell again!")




---


#**Chatbot code ends here**



---



**Step 7.** Log in to https://smith.langchain.com/ to view usage details.



---
---



#**Variations to try**

1. Try different user inputs. For example, you could ask for an explanation for specific functions or lines of code.

2. Alter the system message with "sysmessage". For example, you could try asking for the model to speak informally as if speaking to a friend.

3. Alter the prompt for getting input. For example, you could try something like "I am a friendly technology support assistant. How can I help you?"

**Advanced variations**

1. Alter chatbot to keep a memory of the conversation. This requires feeding into the prompt a history of the conversation. (https://python.langchain.com/docs/how_to/chatbots_memory/)

2. Try different OpenAI models. (https://platform.openai.com/docs/models). The same structure with minimal changes can call other model provider APIs.

3. Create user interface. (https://docs.chainlit.io/integrations/langchain)

