import logging
import os
from datetime import datetime
from pathlib import Path

class Logger:

    def __init__(self,name="EnterpriceRAG"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return

        log_dir = Path("reports/logs")
        log_dir.mkdir(parents=True,exist_ok=True)

        log_file = log_dir / f"rag_{datetime.now():%Y-%m-%d}.log" 
        
        
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            log_file, encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

logger = Logger().get_logger()


