### Prerequisties

Before procceeding, make sure you have the following installed:

1. Python(3.9.6)

2. Docker & docker compose: [docker.com](https://www.docker.com)

- Verify Docker installation

```

docker --version

```

- Verify Docker compose installtion

```

docker-compose --version

```

  

📥 Quick start

- Clone the repo then ```cd Trustpilot/local```

  

### 🐳 Run with Docker compose
This will build and run the API, using the command

```

docker-compose up --build -d

```

The API is running at:

Endpoint: http://127.0.0.1:8000/

Test using Swagger UI: http://127.0.0.1:8000/docs


The swagger UI should look like as follows:
1. Swagger first screen > click on POST
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger1.png?raw=true)
2. In the request body change the value of the text to be of a book review.
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger2.png?raw=true)
3. See the response body showing the sentiment of the text.
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger3.png?raw=true)

### Stop Docker compose

```

docker-compose stop

```
