# 1. Use the official lightweight Python image
FROM python:3.10-slim

# Create a system group and user (matching UID 1000 for Hugging Face compatibility)
RUN groupadd -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup appuser

# 2. Set the working directory inside the container
WORKDIR /app
RUN chown appuser:appgroup /app

# 3. Copy the requirements file and install dependencies
# We use --no-cache-dir to keep the container small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code, data, models, the database, and startup script with proper ownership
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup start.sh /app/start.sh

# Switch to non-root user
USER appuser

# 5. Expose the ports for FastAPI (8000) and Streamlit (8501 / 7860)
EXPOSE 8000
EXPOSE 8501
EXPOSE 7860

# 6. Command to run the entrypoint script starting both services
CMD ["/app/start.sh"]