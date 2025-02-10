import logging
import os

def setup_logger(name=__name__, level=logging.INFO, log_dir="logs", file_name=None):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, file_name)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers(): # Remove existing handlers if any, so that the logger gets reconfigured.
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
