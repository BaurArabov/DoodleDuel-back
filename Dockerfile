FROM python:3.9 as backend

WORKDIR /server

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/datasets .

COPY . .
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]