
"""
author: Vijay
email: uservjkumar@gmail.com
"""


from utils.model import Perceptron
import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_plot, save_model
import logging
import os

logging_str = "[%(asctime)s - %(levelname)s: %(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,
filemode='a')

def main(data, eta ,epochs,filename,plotfilename):

    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe {df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss() ##dummy variable

    save_model(model,filename=filename)
    save_plot(df,plotfilename,model)

if __name__ == '__main__':  ## << entry point
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA=0.3
    EPOCHS=100
    try:
        logging.info(">>>>>>>>>>>>>>>> Starting Training Here >>>>>>>>>>>>>>>>>>>")
        main(OR,ETA,EPOCHS,filename="or.model",plotfilename="or.png")
        logging.info(">>>>>>>>>>>>>>>> training done successfully >>>>>>>>>>>>>>>>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
