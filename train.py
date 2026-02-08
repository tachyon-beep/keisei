# train.py: Thin shim to call the real trainer in keisei.training.train

import asyncio
import multiprocessing

from dotenv import load_dotenv
from keisei.training.train import main

load_dotenv()  # Load environment variables from .env file

if __name__ == "__main__":
    multiprocessing.freeze_support()
    asyncio.run(main())
