# 1. Use the official lightweight Python image
FROM python:3.10-slim

# Create a system group and user
RUN groupadd -g 999 appgroup && \
    useradd -r -u 999 -g appgroup appuser

# 2. Set the working directory inside the container
WORKDIR /app
RUN chown appuser:appgroup /app

# 3. Copy the requirements file and install dependencies
# We use --no-cache-dir to keep the container small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code, data, models, and the database with proper ownership
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup dicra_csi.db /app/dicra_csi.db

# Switch to non-root user
USER appuser

# 5. Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# 6. Command to run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]