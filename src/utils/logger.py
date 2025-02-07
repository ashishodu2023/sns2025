import logging

class Logger:
    def __init__(self, log_file="logs/model.log"):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

    def log(self, message, level="info"):
        if level == "info":
            logging.info(message)
        elif level == "error":
            logging.error(message)
        print(message)

logger = Logger()
