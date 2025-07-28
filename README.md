# 🚀 Agentic Data Assistant

**Transform raw CSV data into actionable business insights with AI-powered analysis**

![Agentic Data Assistant](https://img.shields.io/badge/Agentic-Data%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple)

## 🎯 Overview

The **Agentic Data Assistant** is a multi-agent AI system that transforms raw CSV data into actionable business insights through intelligent analysis, automated visualization, and domain-specific reasoning.

### 🤖 Multi-Agent Architecture
- **📊 Statistical Analysis Agent** - Advanced statistical modeling with ML
- **🔍 Primary Insights Agent** - LangGraph-powered structured analysis  
- **🎯 Domain Insights Agent** - Industry-specific analysis with predictive ML
- **🧠 Deep-Dive Agent** - Tree-of-Thought reasoning with expert perspectives

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API Key
- Docker (optional, for containerized deployment)

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd "Agentic Data Assistant"
```

### 2. Environment Setup
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to access the application.

## 🐳 Deployment Options

### Option 1: Docker (Recommended)
```bash
# Build and run with Docker
docker build -t agentic-data-assistant .
docker run -d -p 8501:8501 --env-file .env agentic-data-assistant
```

### Option 2: Docker Compose (with Redis)
```bash
# Deploy with enhanced caching
docker-compose up -d
```

### Option 3: Automated Deployment Script
```bash
# Run the deployment script
./deploy.sh
```

### Option 4: Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
git push heroku main
```

#### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway up
```

#### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set environment variables in the dashboard
4. Deploy!

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Agent Router   │───▶│  Specialized    │
│   (Frontend)    │    │  (Orchestrator) │    │  AI Agents      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cache Layer   │    │ Data Preprocessor│    │ Chart Generator │
│   (Performance) │    │   (Cleaning)    │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### Frontend
- **Streamlit** - Interactive web interface
- **Custom CSS** - Professional styling

### AI/ML
- **OpenAI GPT-4o** - Primary LLM
- **LangChain** - LLM orchestration
- **LangGraph** - Multi-step workflows
- **Scikit-learn** - Machine learning
- **SciPy** - Statistical analysis

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Visualization

### Performance
- **File-based Caching** - Result caching
- **Performance Monitoring** - Execution tracking

## 📊 Features

### 🧠 Intelligent Analysis
- **Automated Data Profiling** - LLM-powered schema detection
- **Domain Detection** - Industry-specific analysis
- **Statistical Depth** - Correlation, clustering, anomaly detection
- **Predictive Insights** - ML-based forecasting

### 📈 Automated Visualization
- **Smart Chart Selection** - AI-optimized visualizations
- **Interactive Explanations** - "What this chart tells us"
- **Domain-Specific Charts** - Industry-relevant insights
- **Multi-format Output** - PNG with descriptions

### 🔄 Multi-Agent Workflows
- **Agent Selection** - Choose analysis approach
- **Specialized Prompts** - Domain-specific reasoning
- **Tree-of-Thought** - Multi-expert perspectives
- **Iterative Refinement** - Continuous improvement

### ⚡ Performance
- **Intelligent Caching** - Avoid redundant computations
- **Parallel Processing** - Multi-agent execution
- **Memory Management** - Efficient resource usage
- **Real-time Monitoring** - Performance tracking

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration
Edit `.streamlit/config.toml` for custom settings:
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
maxUploadSize = 200
```

## 📁 Project Structure
```
Agentic Data Assistant/
├── app.py                 # Main Streamlit application
├── agents/               # AI agent implementations
│   ├── agent_runner.py   # Agent orchestration
│   ├── enhanced_analysis_agent.py
│   ├── d2insight_agent_sys.py
│   ├── insight2dashboard_tot.py
│   ├── cache_manager.py  # Caching system
│   └── data_preprocessor.py
├── dataset/              # Sample datasets
├── cache/                # Cache storage
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service deployment
├── deploy.sh            # Automated deployment script
└── README.md            # This file
```

## 🚀 Production Deployment

### Docker Production
```bash
# Build optimized image
docker build -t agentic-data-assistant:latest .

# Run with production settings
docker run -d \
  --name agentic-data-assistant \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  agentic-data-assistant:latest
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-data-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-data-assistant
  template:
    metadata:
      labels:
        app: agentic-data-assistant
    spec:
      containers:
      - name: app
        image: agentic-data-assistant:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## 🔒 Security Considerations

### Environment Variables
- Store sensitive keys in environment variables
- Use `.env` files for local development
- Use cloud provider secrets for production

### API Security
- Implement rate limiting
- Add authentication for production use
- Use HTTPS in production

### Data Privacy
- Implement data retention policies
- Clear cache regularly
- Consider data encryption

## 📈 Monitoring & Logging

### Health Checks
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# View logs
docker logs agentic-data-assistant
```

### Performance Monitoring
- Cache hit rates
- Agent execution times
- Memory usage
- API response times

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the code comments
- **Community**: Join our discussions

---

**Made with ❤️ by the Agentic Data Assistant Team**
