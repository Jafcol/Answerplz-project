# GPT2 web service

This is a server built with [Starlette](https://www.starlette.io/ "Starlette") and [Uvicorn](https://www.uvicorn.org/ "Uvicorn") where two GPT2 models are deployed in order to generate answers for Answerplz users.
I use [Docker](https://www.docker.com/ "Docker") to containerize the app and deploy it on Kubernetes.

The main elements of this service are 3:

1. [ONNX](https://onnx.ai/ "ONNX") format that represent GPT2 models to maximize performance across hardware

2. [Spacy](https://spacy.io/ "Spacy") models to process user input and tag entities in it

3. Google service to [translate](https://cloud.google.com/translate "Google cloud translate") generated answers in different languages

To deploy this service you need to:

1. Add your json key to root directory (get it from google cloud platform) and point to the file from Dockerfile environment variable

2. Convert your trained GPT models to ONNX format and put them in root directory ([conversion guide](https://github.com/onnx/tutorials#converting-to-onnx-format "ONNX conversion tutorial"))

3. Train spacy models ([training guide](https://spacy.io/usage/training "Training spacy")) for the five languages supported by the service and add them to root directory

4. Build docker image of this service

5. Deploy containerized app on a platform for managing workloads and services like Kubernetes
