FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
EXPOSE 80
CMD ["python", "fast_api_main.py"]
