import pandas as pd
import config
from sklearn import model_selection
if __name__=="__main__":
    df=pd.read_csv(config.TRAINING_VALUES)
    df1=pd.read_csv(config.TRAINING_LABELS)
    df["labels"]=df1.damage_grade
    
    df['kfold']=-1
    df=df.sample(frac=1).reset_index(drop=True)
    y=df.labels.values
    kf=model_selection.StratifiedKFold(n_splits=5)
    for f, (t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold']=f
    df.to_csv(config.TRAINING_PATH+"train_values_kfold.csv",index=False)
