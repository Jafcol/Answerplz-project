# Scraper

This is a web server built with Puppeteer and MongoDB in order to store user feedback and scrape text from web communities.
I use [Docker](https://www.docker.com/ "Docker") to containerize the app and deploy it on Kubernetes.

[Reddit](https://www.reddit.com/ "Reddit"), [Stack](https://stackexchange.com/ "Stack"), [Fluther](https://www.fluther.com/ "Fluther") and [Answers](https://www.answers.com/ "Answers") are the Q&A platforms from which I retrieve training data for GPT2 models.

To fetch data from Reddit and Stack I'm using their API because Puppeteer's chrome instance can be detected and blocked.

To avoid being blocked during Fluther and Answers scraping process I use some precautions:
1. List of proxies provided by [Webshare](https://www.webshare.io/ "Webshare") in order to rotate IP

2. [Plugin](https://chrome.google.com/webstore/detail/i-dont-care-about-cookies/fihnjjcciajhdojfnbdddfaoknhalnja "I don't care about cookies") to avoid cookie consent popups

3. Service for automatic captcha resolution provided by [2catpcha](https://2captcha.com/ "2catpcha")

4. Puppeteer plugin for a less detectable chrome instance called "[puppeteer-extra-plugin-stealth](https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth "puppeteer plugin stealth")"

Before starting the training process you need to clean scraped data and the best way to do it is using Regex.

The topics that I chose for my project are:

1. english language
2. medical sciences
3. travel
4. cooking
5. gaming
6. movies
7. music
8. scifi
9. pets
10. technology
11. generic asking
12. ask about sex
13. literature
14. history
15. sports
16. finance
17. philosophy
18. mythology
19. science

you can choose other topics, just check the platforms for the right name of the argument.

All the services above need authentication keys that you can obtain visiting their websites.

To run this app you need to:

1. Add your credentials to the code

2. Pull MongoDB image from docker hub and deploy it

3. Build docker image of this service

4. Run the container with this image but remember to mount one of your local folder (where scraped data will be collected as csv files) to "/data"