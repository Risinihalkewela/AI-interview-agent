AI Interview Coach + CV Optimization Agent 🎯  
A true autonomous AI agent built in Python using Streamlit and the Google Gemini API.  
This system goes beyond a chatbot — it perceives, reasons, acts, and learns to maximize interview readiness over time.

🚀 Features
Job Module

Input: company, role, level, responsibilities, requirements

Output: company summary + 20+ HR & technical questions

CV Module

Upload PDF/DOCX CV

Extract text, analyze vs job description

Identify strengths, weaknesses, missing skills, ATS issues

Rewrite CV (ATS‑optimized, aligned to job, no fabricated experience)

Download as PDF/DOCX

Interview Module (Adaptive)

Dynamic questions based on weak skills & past scores

Structured feedback: strengths, weaknesses, improvements

Scores (Technical, Communication, Confidence)

Difficulty escalates as performance improves

Decision Logic

Explicit rules (e.g., if technical score < 3 → more technical questions)

CV gaps seed interview questions

Avoids repeating similar questions

Memory System

Stores weak skills, scores, CV gaps, question history

Learns from every answer → adapts future sessions

Dashboard

Progress bars, overall score, performance trends

Category‑wise charts, weak areas, full Q&A review

One‑click PDF report

Chatbot

Context‑aware career prep advice based on job + CV

🧠 Agent Architecture
Perceive → job inputs, CV, answers, past scores

Reason → detect gaps, identify weak areas, decide next action

Act → generate questions, evaluate answers, rewrite CV, produce reports

Learn → update memory, adapt future behavior, avoid repetition

📂 Tech Stack
Python (single .py file, modular class design)

Streamlit (UI with tabs: Job Input, CV Analysis, Interview, Dashboard)

Google Gemini API (via .env key)

pdfplumber, python-docx (CV parsing)

reportlab (PDF export)

matplotlib / plotly (visualizations)

📌 Why This Is a True AI Agent
Has a goal: maximize interview readiness

Uses memory: stores and adapts based on past sessions

Shows reasoning: explicit decision rules visible to the user

Demonstrates adaptation: difficulty and topics change dynamically

Implements learning: future behavior evolves with every answer
