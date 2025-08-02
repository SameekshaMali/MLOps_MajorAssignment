# Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Train model and quantize it
RUN python src/train.py && python src/quantize.py

# Default CMD runs predict.py to verify container works
CMD ["python", "src/predict.py"]
