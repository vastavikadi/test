# Use the official Python image with the required version
FROM python:3.12

# Set a working directory inside the container
WORKDIR /app

# Copy only the requirements file initially to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies from requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire application into the container's /app directory
COPY . .

# Set environment variables (example: loading .env file contents into environment variables)
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "chatbot.py"]
