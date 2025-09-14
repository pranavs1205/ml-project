# Stage 1: Build stage with dependencies
FROM python:3.9-slim as builder

WORKDIR /app

# Install pipreqs to generate a precise requirements.txt
RUN pip install pipreqs

# Copy only the source code to generate requirements
COPY ./src /app/src
COPY ./app /app/app

# Generate requirements.txt based on actual imports
RUN pipreqs . --force

# Install dependencies into a temporary location
RUN pip install --prefix=/install -r requirements.txt


# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the entire project
COPY . .

# Expose the ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Create a startup script to run both services
RUN echo '#!/bin/bash' > start.sh && \
    echo '# Start the FastAPI backend in the background' > start.sh && \
    echo 'uvicorn app.backend:app --host 0.0.0.0 --port 8000 &' >> start.sh && \
    echo '# Start the Streamlit frontend' >> start.sh && \
    echo 'streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none' >> start.sh && \
    chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]