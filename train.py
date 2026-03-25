# train.py: Thin shim to call the real trainer in keisei.training.train

from dotenv import load_dotenv
from keisei.training.train import main_sync

load_dotenv()  # Load environment variables from .env file

if __name__ == "__main__":
    main_sync()
