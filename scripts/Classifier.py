from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import transformers
import datasets
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

class Classifier():
    def __init__(self):
        pass
    
    def train_test_split(self,preprocessed_data_path="data/pre_processed_data.csv"):
        data=pd.read_csv(preprocessed_data_path)
        label=data["LABEL"].astype("float")
        data.drop(["LABEL"],inplace=True,axis=1)
        X = data  # Features (text data)
        y = label  # Target variable

        # Split the data into training and testing sets (e.g., 70% train, 15% development, 15% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        pd.concat([X_train,y_train],axis=1).to_csv("data/train.csv",index=False)
        pd.concat([X_test,y_test],axis=1).to_csv("data/test.csv",index=False)
        pd.concat([X_dev,y_dev],axis=1).to_csv("data/dev.csv",index=False)

    def tokenize(self,examples):
        #tokenizing using bert
        return tokenizer(examples["TEXT"], truncation=True, max_length=64,
                        padding="max_length",return_tensors="tf")
    

    def fit(self, train_path="data/train.csv", devlopment_path="data/dev.csv"):
        # load the CSVs into Huggingface datasets
        self.humorous_headlines_dataset = datasets.load_dataset("csv", data_files={
            "train": train_path, "validation": devlopment_path})
        #traines the model using bidirectional neural network layer and saves the model into a model file
        self.dataset = self.dataset.map(self.tokenize, batched=True)

        train = self.dataset["train"].to_tf_dataset(
            columns="input_ids",
            label_cols="LABEL",
            batch_size=16,
            shuffle=True)
        devlopment = self.dataset["validation"].to_tf_dataset(
            columns="input_ids",
            label_cols="LABEL",
            batch_size=16)
        early_stop=tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            restore_best_weights=True
        )
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,4 ,embeddings_initializer="RandomUniform"))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(4)))
        model.add(tf.keras.layers.Dense(4))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
                1,
                activation='sigmoid'
                ))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[tf.keras.metrics.F1Score(threshold=0.5)])
        # fit the model to the training data, monitoring F1 on the dev data
        model.fit(
            train,
            epochs=10,
            batch_size=16,
            validation_data=devlopment,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="model",
                    monitor="val_f1_score",
                    mode="max",
                    save_best_only=True),early_stop],
                    class_weight={0:0.89,
                                1:2.079})
    
    def save_to_file(self,path="data/predict_scores.csv"):
        #takes the data makes predicions and saves them into a predictions.csv file 
        # to produce file to upload in kaggle competition
        # loading saved model
        test_data = pd.read_csv(path)
        predictions=self.predict(path)
        predictions=predictions.reshape((1,-1))[0]
        pd.concat([test_data[["ID"]],pd.DataFrame({"LABEL":predictions})],axis=1).to_csv("data/predictions.csv")

    def predict(self,path="data/test.csv"):
        model = tf.keras.models.load_model("model")

        # load the data for prediction
        test_data = pd.read_csv(path)

        # create input features in the same way as in train()
        self.dataset = datasets.Dataset.from_pandas(test_data)
        self.dataset = self.dataset.map(self.tokenize, batched=True)
        
        tf_dataset = self.dataset.to_tf_dataset(
            columns="input_ids",
            batch_size=16)
        
        return numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)
    
    def confusion_matrix(self,path="data/test.csv"):
        test_data=pd.read_csv(path)
        predictions=self.predict(path)
        print(multilabel_confusion_matrix(predictions,test_data[["LABEL"]]))