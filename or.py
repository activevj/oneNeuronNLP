from utils.model import Perceptron
import pandas as pd
import numpy as np
from utils.all_utils import prepare_data, save_plot, save_model

def main(data, eta ,epochs,filename,plotfilename):

    df = pd.DataFrame(data)
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
    EPOCHS=10

    main(OR,ETA,EPOCHS,filename="or.model",plotfilename="or.png")
