# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency files
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application's source code from the current directory to the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py"]
