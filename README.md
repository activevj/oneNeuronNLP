# oneNeuronNLP
oneNeuron | perceptron


```bash
git add . && git commit -m "Update string" && git push origin main
```


## Add URL -
[Git handbook](https://github.com/introduction/git-handbook/)

## You can use Anchor tag 
<a></a>

##Add Image -
![Sample Image](plots/and.png)


## Python code 

```python
def main(data, eta ,epochs,filename,plotfilename):

    df = pd.DataFrame(data)
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss() ##dummy variable

    save_model(model,filename=filename)
    save_plot(df,plotfilename,model)

if __name__ == '__main__':  ## << entry point
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
    }
    ETA=0.3
    EPOCHS=10

    main(AND,ETA,EPOCHS,filename="and.model",plotfilename="and.png")

```
