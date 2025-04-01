# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and app code to container
COPY requirements.txt requirements.txt
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the app with Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "6"]
