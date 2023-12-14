from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import transformers
import datasets
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from nltk import pos_tag
from nltk.tokenize import word_tokenize

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
    
    def split(sef, text):
        a, b=text["TEXT"].split("[SNIPPET]")
        return {"one":a,"two":b}

    def tokenize(self,examples):
        #tokenizing using bert
        return tokenizer(examples["TEXT"], truncation=True, max_length=64,
                        padding="max_length",return_tensors="tf")
    def count_punctuation(self, example):
        punctuation = ["?", ",", ";", ".", ":", "'", '"', "!", "-"]
        
        # Count punctuation for 'one' column
        one_text = example['one']
        one_punctuation_count = {f'one_{punct}_count': one_text.count(punct) for punct in punctuation}
        
        # Count punctuation for 'two' column
        two_text = example['two']
        two_punctuation_count = {f'two_{punct}_count': two_text.count(punct) for punct in punctuation}
        # Merge counts into example dictionary
        example.update({"ONE":one_punctuation_count.values()})
        example.update({"TWO":two_punctuation_count.values()})
        
        return example
    def get_pos_tags(self, sentence):
        tokens = word_tokenize(sentence["one"])
        pos_tags = pos_tag(tokens)
        one_pos=' '.join(t[1] for t in pos_tags)
        tokens = word_tokenize(sentence["two"])
        pos_tags = pos_tag(tokens)
        two_pos=' '.join(t[1] for t in pos_tags)
        limiters = ["DT", "IN", "CC","PRD","MD"]
        one_punctuation_count = {f'one_{punct}_count': one_pos.count(punct) for punct in limiters}
        two_punctuation_count = {f'one_{punct}_count': two_pos.count(punct) for punct in limiters}
        sentence.update({"one_pos":one_punctuation_count.values()})
        sentence.update({"two_pos":two_punctuation_count.values()})
        return sentence

    

    def fit(self, train_path="data/train.csv", devlopment_path="data/dev.csv"):
        # load the CSVs into Huggingface datasets
        self.dataset = datasets.load_dataset("csv", data_files={
            "train": train_path, "validation": devlopment_path})
        #traines the model using bidirectional neural network layer and saves the model into a model file
        self.dataset = self.dataset.map(self.tokenize, batched=True)
        self.dataset = self.dataset.map(self.split)
        self.dataset = self.dataset.map(self.count_punctuation)
        self.dataset = self.dataset.map(self.get_pos_tags)
        pos=tf.convert_to_tensor(self.dataset["train"]["ONE"])
        pso=tf.convert_to_tensor(self.dataset["train"]["TWO"])
        text=tf.convert_to_tensor(self.dataset["train"]["input_ids"])
        label=tf.expand_dims(tf.convert_to_tensor(self.dataset["train"]["LABEL"]), axis=1)
        label_train=tf.cast(
        label, dtype=tf.float32)
        train_dataset=tf.concat([text, pos, pso, tf.convert_to_tensor(self.dataset["train"]["one_pos"]), tf.convert_to_tensor(self.dataset["train"]["two_pos"])],1)
        pos=tf.convert_to_tensor(self.dataset["validation"]["ONE"])
        pso=tf.convert_to_tensor(self.dataset["validation"]["TWO"])
        text=tf.convert_to_tensor(self.dataset["validation"]["input_ids"])
        label=tf.expand_dims(tf.convert_to_tensor(self.dataset["validation"]["LABEL"]), axis=1)
        label_dev=tf.cast(
        label, dtype=tf.float32)
        dev_dataset=tf.concat([text, pos, pso, tf.convert_to_tensor(self.dataset["validation"]["one_pos"]), tf.convert_to_tensor(self.dataset["validation"]["two_pos"])],1)
        # convert Huggingface datasets to Tensorflow datasets
        early_stop=tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            patience=2,
            mode="max"
        )
        #initializer = tf.keras.initializers.RandomNormal(mean=0.,stddev=0.01)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,8))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(8, return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(8)))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
                1,
                activation='sigmoid'
                ))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[tf.keras.metrics.F1Score(threshold=0.5)])
        # fit the model to the training data, monitoring F1 on the dev data
        model.fit(
            train_dataset,
            label_train,
            epochs=10,
            batch_size=16,
            validation_data=(dev_dataset,label_dev),
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
        pd.concat([test_data[["ID"]],pd.DataFrame({"LABEL":predictions})],axis=1).to_csv("data/predictions.csv", index=False)

    def predict(self,path="data/test.csv"):
        model = tf.keras.models.load_model("model")

        # load the data for prediction
        test_data = pd.read_csv(path)

        # create input features in the same way as in train()
        self.dataset = datasets.Dataset.from_pandas(test_data)
        self.dataset = self.dataset.map(self.tokenize, batched=True)
        self.dataset = self.dataset.map(self.split)
        self.dataset = self.dataset.map(self.count_punctuation)
        self.dataset = self.dataset.map(self.get_pos_tags)
        self.dataset = self.dataset.map(self.split)
        self.dataset = self.dataset.map(self.count_punctuation)
        self.dataset = self.dataset.map(self.get_pos_tags)
        pos=tf.convert_to_tensor(self.dataset["ONE"])
        pso=tf.convert_to_tensor(self.dataset["TWO"])
        text=tf.convert_to_tensor(self.dataset["input_ids"])
        tf_dataset=tf.concat([text, pos, pso, tf.convert_to_tensor(self.dataset["one_pos"]), tf.convert_to_tensor(self.dataset["two_pos"])],1)
        
        return numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)
    
    def confusion_matrix(self,path="data/test.csv"):
        test_data=pd.read_csv(path)
        predictions=self.predict(path)
        print(confusion_matrix(predictions,test_data[["LABEL"]]))