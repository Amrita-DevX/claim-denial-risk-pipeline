@echo off
cd C:\Users\amrit\PythonPractice\Machine learning\claim-denial-risk-pipeline
venv\Scripts\python.exe src\batch_score.py >> logs\batch_run.log 2>&1