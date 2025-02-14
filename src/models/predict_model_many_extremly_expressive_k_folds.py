import os
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from models.predict_model_wrapper import PredictModelWrapper

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"



if __name__ == "__main__": 

    PredictModelWrapper.predict(
        model_name="best_model_w1_extreme_exposed.pt",
        csv_name="weight_one_extremly_exposed_submission.csv"
    )   
