import mlflow.artifacts
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
import sqlite3
import dagshub
# from mlflow import sklearn
import mlflow


#Intilize the Dagshub and Mlflow tracking
dagshub.init(repo_owner="pradeepradee1",repo_name="MLOPS",mlflow=True)
mlflow.set_experiment("Experimnet2")
mlflow.set_tracking_uri("https://dagshub.com/pradeepradee1/MLOPS.mlflow")

#Data Ingestion
data = pd.read_csv("notebooks\Housing.csv")
print("Records {0} and attributes {1}".format(data.shape[0],data.shape[1]))

#Checking the null values
null_counts = data.isnull().sum()


#Feature Engineering
Target_Feature = ["price"]
Input_Feature = [i for i in list(data.columns) if i not in Target_Feature ]
Dependent_Feature = data[Target_Feature]
Independent_Feature = data[Input_Feature]
le = LabelEncoder()
categoricaldata=Independent_Feature.select_dtypes(include=['object'])
categoricaldata["mainroad"] = le.fit_transform(categoricaldata["mainroad"])
categoricaldata["guestroom"] = le.fit_transform(categoricaldata["guestroom"])
categoricaldata["basement"] = le.fit_transform(categoricaldata["basement"])
categoricaldata["hotwaterheating"] = le.fit_transform(categoricaldata["hotwaterheating"])
categoricaldata["airconditioning"] = le.fit_transform(categoricaldata["airconditioning"])
categoricaldata["prefarea"] = le.fit_transform(categoricaldata["prefarea"])
categoricaldata["furnishingstatus"] = le.fit_transform(categoricaldata["furnishingstatus"])
#categoricaldata.head(5)
continousdata=Independent_Feature.select_dtypes(exclude=['object'])
#continousdata.head(5)

sampledataframe=pd.concat([continousdata,categoricaldata],axis=1)
# sampledataframe.head(5)
sampledataframe["price"]=Dependent_Feature

Columns=list(sampledataframe.columns)
# print(list(sampledataframe.values))

#Connecting With DB
conn = sqlite3.connect('Features_Storage.db')
sampledataframe.to_sql('Features', conn, if_exists='replace', index=False)

cur = conn.cursor()
cur.execute("SELECT * FROM Features")
rows = cur.fetchall()

print(type(rows))


for row in rows:
    print(row)
    break

conn.close()

sampledataframe=pd.DataFrame(rows)
sampledataframe.columns=Columns

X, y = sampledataframe.drop('price', axis=1), sampledataframe[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


model = { "LinearRegression":LinearRegression(),
         "RandomForestRegressor" : RandomForestRegressor(n_estimators=100, random_state=42)}

with mlflow.start_run(run_name="House Price Predictions"):

    for model_name,model in model.items():
        with mlflow.start_run(run_name=model_name,nested=True):
            model.fit(X_train,y_train)

            model_filename = f"{model_name.replace(' ', '_')}.pkl"

            pickle.dump(model,open(model_filename,"wb"))

            predictions = model.predict(X_test)

            mae = mean_absolute_error(predictions, y_test)
            r2 = r2_score(predictions, y_test)

            mlflow.log_metric("MAE",mae)
            mlflow.log_metric("r2",r2)
            print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

            mlflow.sklearn.log_model(model,model_name.replace(" ","_"))
            
            mlflow.log_artifact(__file__)

            mlflow.set_tag("author","pradeepradee1")
            mlflow.set_tag("model","GB")

            print("Mean Absolute Error {}".format(mae))
            print("R-squared:", r2)
