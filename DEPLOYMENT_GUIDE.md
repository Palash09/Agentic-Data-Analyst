# 🚀 Deployment Guide - Agentic Data Assistant

This guide provides comprehensive deployment options for the Agentic Data Assistant, from local development to production cloud deployment.

## 📋 Prerequisites

### Required
- **Python 3.11+** - Core runtime
- **OpenAI API Key** - For AI functionality
- **Git** - Version control

### Optional (for specific deployments)
- **Docker** - Containerized deployment
- **Heroku CLI** - Heroku deployment
- **Railway CLI** - Railway deployment
- **Kubernetes** - Enterprise deployment

## 🔧 Environment Setup

### 1. Create Environment File
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt
```

## 🏠 Local Development

### Quick Start
```bash
# Run the application locally
streamlit run app.py
```

### Production-like Local
```bash
# Run with production settings
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

**Access:** http://localhost:8501

## 🐳 Docker Deployment

### Option 1: Simple Docker
```bash
# Build the image
docker build -t agentic-data-assistant .

# Run the container
docker run -d \
  --name agentic-data-assistant \
  -p 8501:8501 \
  --env-file .env \
  agentic-data-assistant
```

### Option 2: Docker Compose (Recommended)
```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 3: Production Docker
```bash
# Build optimized image
docker build -t agentic-data-assistant:latest .

# Run with production settings
docker run -d \
  --name agentic-data-assistant \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/dataset:/app/dataset \
  agentic-data-assistant:latest
```

## ☁️ Cloud Deployment

### Heroku Deployment

#### 1. Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku

# Windows
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

#### 2. Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_openai_api_key

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open the app
heroku open
```

#### 3. Heroku Configuration
```bash
# Scale the app
heroku ps:scale web=1

# View logs
heroku logs --tail

# Check app status
heroku ps
```

### Railway Deployment

#### 1. Install Railway CLI
```bash
npm install -g @railway/cli
```

#### 2. Deploy to Railway
```bash
# Login to Railway
railway login

# Initialize project
railway init

# Set environment variables
railway variables set OPENAI_API_KEY=your_openai_api_key

# Deploy
railway up

# Get deployment URL
railway domain
```

### Streamlit Cloud Deployment

#### 1. Prepare Repository
```bash
# Ensure your code is on GitHub
git add .
git commit -m "Prepare for Streamlit Cloud"
git push origin main
```

#### 2. Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set the path to `app.py`
6. Add environment variables:
   - `OPENAI_API_KEY`: your_api_key
7. Click "Deploy"

## 🏢 Enterprise Deployment

### Kubernetes Deployment

#### 1. Create Namespace
```bash
kubectl create namespace agentic-data
```

#### 2. Create Secret
```bash
kubectl create secret generic openai-secret \
  --from-literal=api-key=your_openai_api_key \
  -n agentic-data
```

#### 3. Deploy Application
```bash
# Apply the deployment
kubectl apply -f k8s-deployment.yaml -n agentic-data

# Check status
kubectl get pods -n agentic-data
kubectl get services -n agentic-data
```

#### 4. Kubernetes Manifests
Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-data-assistant
  namespace: agentic-data
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-data-service
  namespace: agentic-data
spec:
  selector:
    app: agentic-data-assistant
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### AWS ECS Deployment

#### 1. Create ECR Repository
```bash
aws ecr create-repository --repository-name agentic-data-assistant
```

#### 2. Build and Push Image
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t agentic-data-assistant .
docker tag agentic-data-assistant:latest your-account.dkr.ecr.us-east-1.amazonaws.com/agentic-data-assistant:latest

# Push to ECR
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/agentic-data-assistant:latest
```

#### 3. Deploy to ECS
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name agentic-data-cluster

# Create task definition and service
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cli-input-json file://service-definition.json
```

## 🔒 Security Configuration

### Environment Variables
```bash
# Production environment variables
OPENAI_API_KEY=your_openai_api_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

### SSL/HTTPS Setup
```bash
# For production, use a reverse proxy like nginx
# Example nginx configuration:
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint
```bash
# Check application health
curl http://your-domain.com/_stcore/health
```

### Logging
```bash
# Docker logs
docker logs agentic-data-assistant

# Kubernetes logs
kubectl logs -f deployment/agentic-data-assistant -n agentic-data

# Heroku logs
heroku logs --tail
```

### Performance Monitoring
- **Cache hit rates** - Monitor cache effectiveness
- **Agent execution times** - Track performance
- **Memory usage** - Monitor resource consumption
- **API response times** - Track OpenAI API performance

## 🚀 Automated Deployment

### Using the Deployment Script
```bash
# Make script executable
chmod +x deploy.sh

# Run deployment script
./deploy.sh
```

The script will guide you through:
1. Environment validation
2. Deployment option selection
3. Automated deployment process
4. Health check verification

### CI/CD Pipeline (GitHub Actions)
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy Agentic Data Assistant

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
        heroku_app_name: ${{ secrets.HEROKU_APP_NAME }}
        heroku_email: ${{ secrets.HEROKU_EMAIL }}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenAI API Key Error
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Verify in container
docker exec agentic-data-assistant env | grep OPENAI
```

#### 2. Port Already in Use
```bash
# Check what's using port 8501
lsof -i :8501

# Kill process or use different port
docker run -p 8502:8501 agentic-data-assistant
```

#### 3. Memory Issues
```bash
# Increase memory limits
docker run --memory=2g agentic-data-assistant

# Or in docker-compose.yml
services:
  agentic-data-assistant:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### 4. Cache Issues
```bash
# Clear cache
rm -rf cache/*

# Rebuild Docker image
docker build --no-cache -t agentic-data-assistant .
```

## 📈 Scaling Considerations

### Horizontal Scaling
```bash
# Docker Compose scaling
docker-compose up --scale agentic-data-assistant=3

# Kubernetes scaling
kubectl scale deployment agentic-data-assistant --replicas=5
```

### Load Balancing
- Use nginx or HAProxy for load balancing
- Implement sticky sessions if needed
- Monitor resource usage across instances

### Database Integration
- Consider adding PostgreSQL for persistent storage
- Implement Redis for enhanced caching
- Add monitoring with Prometheus/Grafana

## 🎯 Next Steps

1. **Choose your deployment method** based on your needs
2. **Set up monitoring** for production environments
3. **Implement security measures** (authentication, rate limiting)
4. **Plan for scaling** as your user base grows
5. **Set up backups** for critical data
6. **Document your deployment** for team members

---

**Need help?** Check the main README.md or create an issue on GitHub! 