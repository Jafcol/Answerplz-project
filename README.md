# Answerplz project

Answerplz is an artificial community created by training a neural network on different topics such as science, movies, music and health.
The AI used is called GPT2 and is produced by [OpenAI](https://openai.com/ "OpenAI"), the data used to train it is extracted from the most frequented forums
on the web and is designed to learn from user feedback.
The main elements of this project are 4:

1. Web scraper built with [Puppeteer](https://pptr.dev/ "Puppeteer") and [MongoDB](https://www.mongodb.com/ "MongoDB")

2. [GPT2](https://openai.com/blog/gpt-2-1-5b-release/ "OpenAI's GPT2") models hosted as a service on [Kubernetes](https://kubernetes.io/ "Kubernetes")

3. Android app where users can interact vocally with the AI and leave their feedback

4. Website where users can interact with the AI and leave their feedback