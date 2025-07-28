#!/bin/bash

# Agentic Data Assistant Deployment Script
# This script provides multiple deployment options

set -e

echo "🚀 Agentic Data Assistant Deployment Script"
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo "Example:"
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    exit 1
fi

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    # Build the Docker image
    echo "Building Docker image..."
    docker build -t agentic-data-assistant .
    
    # Run the container
    echo "Starting container..."
    docker run -d \
        --name agentic-data-assistant \
        -p 8501:8501 \
        --env-file .env \
        -v $(pwd)/cache:/app/cache \
        -v $(pwd)/dataset:/app/dataset \
        agentic-data-assistant
    
    echo "✅ Application deployed at http://localhost:8501"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    echo "🐳 Deploying with Docker Compose..."
    
    # Start services
    docker-compose up -d
    
    echo "✅ Application deployed at http://localhost:8501"
    echo "📊 View logs: docker-compose logs -f"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "☁️ Deploying to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "❌ Heroku CLI not found. Please install it first:"
        echo "https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Create Heroku app if it doesn't exist
    if ! heroku apps:info &> /dev/null; then
        echo "Creating new Heroku app..."
        heroku create
    fi
    
    # Set environment variables
    echo "Setting environment variables..."
    heroku config:set OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
    
    # Deploy
    echo "Deploying to Heroku..."
    git add .
    git commit -m "Deploy to Heroku" || true
    git push heroku main
    
    echo "✅ Application deployed to Heroku"
    echo "🌐 URL: $(heroku info -s | grep web_url | cut -d '=' -f2)"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        echo "❌ Railway CLI not found. Please install it first:"
        echo "npm install -g @railway/cli"
        exit 1
    fi
    
    # Login to Railway
    railway login
    
    # Initialize Railway project
    railway init
    
    # Deploy
    railway up
    
    echo "✅ Application deployed to Railway"
}

# Function to deploy to local server
deploy_local() {
    echo "💻 Deploying locally..."
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Start the application
    echo "Starting application..."
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
    
    echo "✅ Application running at http://localhost:8501"
}

# Main menu
echo ""
echo "Choose deployment option:"
echo "1) Docker (Recommended)"
echo "2) Docker Compose (with Redis)"
echo "3) Heroku (Cloud)"
echo "4) Railway (Cloud)"
echo "5) Local Server"
echo "6) Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        deploy_docker
        ;;
    2)
        deploy_docker_compose
        ;;
    3)
        deploy_heroku
        ;;
    4)
        deploy_railway
        ;;
    5)
        deploy_local
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac 