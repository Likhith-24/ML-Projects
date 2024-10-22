import logging, os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) # log file path

logging.basicConfig(
    filename=LOG_FILE_PATH,  # log file name
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # log format
    level=logging.INFO, # log level
    )


if __name__ == "__main__":
    logging.info("This is an info message")