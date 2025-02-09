import logging
import os

def setup_logger(name=__name__, level=logging.INFO, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),        # log to console
            logging.FileHandler(log_file)     # log to file
        ]
    )
    return logging.getLogger(name)

logger = setup_logger(log_dir="logs")  # you can override log_dir if needed