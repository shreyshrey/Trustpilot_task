
## Prerequisties

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

- Open terminal to clone the repo 
- Once cloned then ```cd Trustpilot/local```

  

## 🐳 Run with Docker compose
- Make sure when in the terminal, you are in the `local` folder where the Docker files are.
- Run the below command to build and run the API:

```
docker-compose up --build -d
```

The API should be running at:

Endpoint: http://127.0.0.1:8000/

Test using Swagger UI: http://127.0.0.1:8000/docs


The swagger UI should look like as follows:
1. Swagger first screen > click on POST /predict/ and hit "Try it out".
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger1.png?raw=true)
2. In the request body change the value of the text to be of a book review.
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger2.png?raw=true)
3. See the response body showing the sentiment of the text.
![](https://github.com/shreyshrey/Trustpilot_task/blob/master/swagger3.png?raw=true)


### To check the p99 latency for model predictions
``` pip install hey ``` or ```brew install hey```

#### Run load test
Make sure the image was deployed and the API is running. Then paste the following command:
```
hey -n 1000 -c 50 -m POST -H "Content-Type: application/json" -d '{"text": "This book was very informative and enjoyable."}' http://127.0.0.1:8000/predict/
```
This will give the:
- Send 1000 POST requests with 50 concurrent users.
- Display detailed latency metrics, including p99.

## Stop Docker compose
Once testing is done, stop the container using the following command:
```

docker-compose stop

```

## To run without docker

### 🐍 Set up virtual environment

Open terminal and type/copy the following command:
``` bash
python -m venv sentiment_env

# Activate the env
source sentiment_env/bin/activate # on Linux/Max

sentiment_env\Scripts\activate # on windows
```

### 📦 Install Dependencies
```bash
pip install -r requirments.txt
```

### 🧠 Train and save the model
```bash
python model/train_model.py
```
- This will save the model to `./model_output/sentiment_model.pth` and vectorised to `./model_output/vectorizer.npy` and label encoder to `./model_output/label_encoder.npy`.

### Run the FastAPI application

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --workers 4
```
- Open another terminal to test the API using the command:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{"text": "This book was absolutely fantastic! Highly recommended."}'
```



