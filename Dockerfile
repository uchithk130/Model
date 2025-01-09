FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD /bin/sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
