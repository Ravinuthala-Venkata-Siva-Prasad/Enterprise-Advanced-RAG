from pathlib import Path
from src.utils.logger import logger
from src.utils.config_loader import ConfigLoader

def create_project_directories():

    directories = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "data/evaluation",

        "reports/logs",
        "reports/evaluation",
        "reports/performance",

        "notebooks",
        "tests",
        "deployment"

    ]

    for directory in directories:

        Path(directory).mkdir(parents=True,exist_ok=True)
        logger.info(f"Directory ready: {directory}")

def main():

    try: 
        logger.info("Starting Enterprice Advanced RAG Platform")
        create_project_directories()
        config = ConfigLoader.load_yaml("config/config.yaml")
        model_config = ConfigLoader.load_yaml("config/model_config.yaml")
        logger.info(f"Project: {config['project']['name']}")
        logger.info(f"LLM provider: {model_config['llm']['provider']}")
        logger.info("Project initalization completed successfully")

    except Exception:
        logger.exception("Application initialization failed")
        raise


if __name__=="__main__":
    main()