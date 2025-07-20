FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app app/
COPY model model/

# Create model directory if it doesn't exist and ensure model file is present
RUN if [ ! -f "model/model.pkl" ]; then \
        echo "Warning: model.pkl not found. Training model..." && \
        python model/train.py; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]