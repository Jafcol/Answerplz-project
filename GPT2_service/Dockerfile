FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

WORKDIR /

COPY . .

ENV GOOGLE_APPLICATION_CREDENTIALS=key.json

# Make changes to the requirements/app here.
RUN pip3 --no-cache-dir install starlette uvicorn ujson pathlib onnx onnxruntime-gpu==1.4.0 onnxruntime-tools transformers torch google-cloud os gc psutil numpy typing random cupy-cuda100==7.3.0 spacy[cuda100]==2.3

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "server.py"]