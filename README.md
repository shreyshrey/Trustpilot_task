Prerequisties
Before procceeding, make sure you have the following installed:
1. Python(3.9.6) installed
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
- Clone the repo
cd Trustpilot/local

# Run with Docker compose
```
docker-compose up --build -d
```
This will build and run the API. The API is running at:
Endpoint: http://127.0.0.1:8000/
Test using Swagger UI: http:127.0.0.1:8000/docs

The swagger UI should look like as follows:



# Stop Docker compose
```
docker-compose stop
```
