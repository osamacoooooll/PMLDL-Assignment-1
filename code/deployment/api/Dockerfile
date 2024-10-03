FROM python:3.9-slim

RUN apt-get update && apt-get install -y python3-pip

RUN pip install --upgrade pip

WORKDIR /api

COPY ./deployment/api/ . 

COPY ../../models/model.pkl /api/models/model.pkl

COPY ../../models/scaler.pkl /api/models/scaler.pkl

# here you can use requriment file instead ...
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn==1.5.1 joblib pydantic

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]