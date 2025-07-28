# Render Deployment Guide

## Quick Deploy to Render

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Select repository: Palash09/Agentic-Data-Analyst
5. Configure:
   - Name: agentic-data-assistant
   - Environment: Python
   - Build Command: pip install -r requirements.txt
   - Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
6. Add Environment Variables:
   - PYTHON_VERSION = 3.11.18
   - OPENAI_API_KEY = your-api-key-here
7. Click "Create Web Service"
8. Wait 5-10 minutes for build
9. Access at: https://agentic-data-assistant.onrender.com
