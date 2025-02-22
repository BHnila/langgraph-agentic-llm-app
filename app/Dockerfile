# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY --chown=appuser:appuser ./app /app

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

# Remove unnecessary tools and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit_app.py when the container launches
CMD streamlit run __main__.py