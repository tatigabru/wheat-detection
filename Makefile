build:
	docker build -t wheat:0.1  .

# Start development container with docker compose in deamon mode
start:
	docker-compose up -d

# Install dependencies inside container
install:
	docker exec wheat-dev pip install -r requirements.txt --user

# Stop your services once you’ve finished with them
stop:
	docker-compose stop

# Bring everything down, removing the containers entirely
clean:
	docker-compose down --volumes

# download and unpack dataset
dataset:
	docker exec wheat-dev bash scripts/download_dataset.sh

# Run trainingd of EfficientDet 4 
train:
	docker exec wheat-dev bash scripts/train.sh

# Make prediction for test data (trained models in models/ directory)
inference:
	docker exec wheat-dev bash scripts/predict.sh