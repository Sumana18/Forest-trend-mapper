FROM python:3.10

# Create a user to avoid running as root (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV UVICORN_PROXY_HEADERS=1
ENV FORWARDED_ALLOW_IPS="*"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8765

CMD ["solara", "run", "app_v1.0.0.py", "--host=0.0.0.0", "--port=8765", "--log-level-uvicorn=info"]
