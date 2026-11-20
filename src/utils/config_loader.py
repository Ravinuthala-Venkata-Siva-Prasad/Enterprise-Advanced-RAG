import yaml
from pathlib import Path

from src.utils.logger import logger
from src.utils.exception import EnterpriseRAGException

import sys

class ConfigLoader:

    @staticmethod
    def load_yaml(file_path):
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")

            with open(file_path,'r',encoding='utf-8') as file:
                config = yaml.safe_load(file)

            logger.info(f"Configuration loader: {file_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise EnterpriseRAGException(e,sys) from e