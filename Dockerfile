FROM python:3.9 as backend

WORKDIR /server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "draw_sketch:app", "--host", "0.0.0.0", "--port", "8000"]