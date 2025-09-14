FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install your existing requirements
COPY dev_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r dev_requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting ML Application..."\n\
echo "📊 Starting FastAPI backend..."\n\
uvicorn app.backend:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
echo "🎭 Starting Streamlit frontend..."\n\
streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none --server.enableCORS false\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
