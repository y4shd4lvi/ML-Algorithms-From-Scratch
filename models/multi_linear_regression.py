import pandas as pd
import numpy as np

class MultiLinearRegression:
    def __init__(self):
        self.beta=None
    
    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        
        ones=np.ones((X.shape[0],1))
        X=np.hstack((ones,X))
        
        XT=X.T
        self.beta=np.linalg.inv(XT @ X) @ XT @ y
    
    def predict(self,X):
        X=np.array(X)
        
        ones=np.ones((X.shape[0],1))
        X=np.hstack((ones,X))
        
        return X @ self.beta
                    
df=pd.read_csv(r"C:\Users\yash dalvi\OneDrive\Desktop\Scratch\data\petrol_consumption.csv")   
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

model=MultiLinearRegression()
model.fit(X,y)
print("Training completed. Coefficients:", model.beta)
prediction=model.predict([[7.5,4870,2351,0.529]])
print("Prediction:", prediction)                 
            