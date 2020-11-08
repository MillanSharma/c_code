import joblib
import argparse
import pandas as pd
import numpy as np
import config
import model_dispatcher
from sklearn import metrics
from sklearn import tree
import os

def run(fold, model):
    train_values_df=pd.read_csv(config.TRAINING_VALUES)
    train_labels_df=pd.read_csv(config.TRAINING_LABELS)
    #replacing all the object values with numberical values 
    train_values_df["land_surface_condition"]=train_values_df["land_surface_condition"].replace(["n", "o", "t"],[1,2,3])
    train_values_df["foundation_type"]=train_values_df["foundation_type"].replace(["h","i", "r", "u", "w"],[1,2,3,4,5])
    train_values_df["roof_type"]=train_values_df["roof_type"].replace(["n","q","x"],[1,2,3])
    train_values_df["ground_floor_type"]=train_values_df["ground_floor_type"].replace(["f", "m", "v", "x", "z"],[1,2,3,4,5])
    train_values_df["other_floor_type"]=train_values_df["other_floor_type"].replace(["j","q", "s", "x"],[1,2,3,4])
    train_values_df["position"]=train_values_df["position"].replace(["j","o","s","t"],[1,2,3,4])
    train_values_df["plan_configuration"]=train_values_df["plan_configuration"].replace(["a", "c", "d", "f","m", "n", "o", "q", "s", "u"],[1,2,3,4,5,6,7,8,9,10])
    train_values_df["legal_ownership_status"]=train_values_df["legal_ownership_status"].replace(["a", "r", "v", "w"],[1,2,3,4])
    train_values_df["labels"]=train_labels_df.damage_grade
    
    train_values=train_values_df[train_values_df.kfold != fold].reset_index(drop=True)    
    valid_values=train_values_df[train_values_df.kfold == fold].reset_index(drop=True)
    
    x_train=train_values.drop("labels",axis=1).values
    y_train = train_values.labels.values
    
    x_valid = valid_values.drop('labels',axis=1).values
    y_valid = valid_values.labels.values
    
    clf=model_dispatcher.models[model]
    clf.fit(x_train,y_train)
    preds=clf.predict(x_valid)
    print("datatype of preds is:",type(preds))
    print("size of pred:",len(preds))
    print("one such sample of pred is :",preds[33],preds[35],preds[56])
    accuracy=metrics.f1_score(y_valid,preds,average="micro")
    print(f"Fold={fold}: Acuuracy={accuracy}")
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f'dt_{fold}.bin'))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
    "--fold",
    type=int)
    parser.add_argument(
    "--model",
    type=str)
    args=parser.parse_args()
    run(fold=args.fold,model=args.model)
 #some comments are missing 
 
 #some typos fixed

    