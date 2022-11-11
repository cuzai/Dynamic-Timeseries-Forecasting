import os

import tensorflow as tf
from Pipelines import *
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import (Dense, Embedding, Input, Masking, SimpleRNN, Bidirectional, TimeDistributed, concatenate)

class Architecture():
    def __init__(self, pipe):
        self.pipe = pipe
    
    def get_model(self):
        # numerical layer
        unit_input_layer = Input(shape=(None, 1))
        sinmonth_input_layer = Input(shape=(None, 1))
        cosmonth_input_layer = Input(shape=(None, 1))

        unit_masked_layer = Masking(mask_value=0)(unit_input_layer)
        sinmonth_masked_layer = Masking(mask_value=0)(sinmonth_input_layer)
        cosmonth_masked_layer = Masking(mask_value=0)(cosmonth_input_layer)

        # categorical layer
        region_input_layer = Input(shape=(None,))
        company_input_layer = Input(shape=(None,))
        month_input_layer = Input(shape=(None,))

        region_nclasses = len(self.pipe.named_steps["Encode"].region_encoder_classes_)
        company_nclasses = len(self.pipe.named_steps["Encode"].company_encoder_classes_)
        month_nclasses = len(self.pipe.named_steps["Encode"].month_encoder_classes_)

        region_embedded_layer = Embedding(input_dim=region_nclasses, output_dim=1024, mask_zero=True)(region_input_layer)
        company_embedded_layer = Embedding(input_dim=company_nclasses, output_dim=1024, mask_zero=True)(company_input_layer)
        month_embedded_layer = Embedding(input_dim=month_nclasses, output_dim=1024, mask_zero=True)(month_input_layer)

        # list layers
        input_layer_li = [unit_input_layer, sinmonth_input_layer, cosmonth_input_layer, region_input_layer, company_input_layer, month_input_layer]
        processed_layer_li = [unit_masked_layer, sinmonth_masked_layer, cosmonth_masked_layer, region_embedded_layer, company_embedded_layer, month_embedded_layer]

        # model
        concat = concatenate(processed_layer_li)

        forward = SimpleRNN(1024, return_sequences=True, return_state=True, activation="relu")
        backward = SimpleRNN(1024, return_sequences=True, return_state=True, go_backwards=True)
        rnn = Bidirectional(layer=forward, backward_layer=backward)(concat)

        forward = SimpleRNN(1024, return_sequences=True, return_state=True, activation="relu")
        backward = SimpleRNN(1024, return_sequences=True, return_state=True, go_backwards=True)
        rnn = Bidirectional(layer=forward, backward_layer=backward)(rnn)

        forward = SimpleRNN(1024, return_sequences=True, activation="relu")
        backward = SimpleRNN(1024, return_sequences=True, go_backwards=True)
        rnn = Bidirectional(layer=forward, backward_layer=backward)(rnn)

        dense = TimeDistributed(Dense(1, activation="relu"))(rnn)

        model = tf.keras.models.MOdel(inputs=input_layer_li, outputs=dense)
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
        return model

class Evaluate():
    def __init__(self, phone, df, df_init, pipe, model):
        self.df = df[df["std_phone_nm"] == phone]
        self.df_init = df_init[df_init["std_phone_nm"] == phone]
        self.pipe = pipe
        self.model = model
    
    def get_new_month(self, df):
        # adding up months by one
        month_no = df.loc[df.shape[0]-1, "month_no"] # get month_no from the last row of the dataframe
        year, month = int(month_no[:4]), int(month_no[-2:])

        if month < 12: month += 1
        else : 
            month = 1
            year += 1
        
        return f"{year}{str(month).zfill(2)}"
    
    def predict_observation(self, df, month_len):
        df_result = pd.DataFrame()
        for n, region in enumerate(df["region"].unique()):
            df_region = df[df["region"] == region].reset_index(drop=True)
            df_feed = df_region.iloc[:month_len].reset_index(drop=True)

            for i in range(24):
                feed, _ = self.pipe.transform(df_feed)
                pred = self.model.predict(feed).reshape(-1)[-1]
                if pred <= 0: break

                df_feed = df_feed.append(df_feed.iloc[-1]).reset_index(drop=True)
                df_feed.loc[df_feed.shape[0]-1, "unit"] = pred
                df_feed.loc[df_feed.shape[0]-1, "month_no"] = self.get_new_month(df_feed)
                df_feed.loc[df_feed.shape[0]-1, "order_month"] = df_feed.loc[df_feed.shape[0]-1, "order_month"]+1
                
            df_feed = df_feed[["region", "std_phone_nm", "month_no", "unit"]]
            df_feed["y"] = df_region["unit"]
            df_result = df_feed if n == 0 else pd.concat([df_result, df_feed])
        return df_result
    
    def predict_initial(self):
        df_result = self.predict_observation(self.df_init, 2) # 2 is just a dummy number. it can be any number larger than 1
        df_result = df_result.rename(columns={"unit":"observation0"})
        return df_result[["region", "month_no", "observation0"]]
    
    def predict(self):
        len_obs = 13; len_obs += 1
        df_result = pd.DataFrame()
        for month_len in range(1, len_obs):
            df = self.predict_observation(self.df, month_len)
            df = df.rename(columns={"unit":f"observation{month_len}"})
            df_result = df if month_len == 1 else df_result.merge(df, on=["region", "std_phone_nm", "month_no", "y"], how="outer")
        
        df_initial_predict = self.predict_initial()
        df_result = df_result.merge(df_initial_predict, on=["region", "month_no"], how="outer")
        return df_result[["region", "std_phone_nm", "month_no", "y", *[f"obeservation{i}" for i in range(len_obs)]]]
    
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    df = pd.read_parquet("/home/working/01_Data/01_Raw/df_base_reg_221109.parquet")
    df = df[df["dataset"] == "raw"]

    df_init = pd.read_parquet("/home/working/01_Data/01_Raw/df_init_result_221109.parquet")
    df_init = df_init[df_init["dataset"] == "raw"]
    df_init = df_init.rename(columns={"sim_pred_unit":"unit"})
    df_init["order_month"] = 1

    pipe = Pipeline([
        ("FeatureEngineer", FeatureEngineer()),
        ("Encode", Encode()),
        ("GetSeq", GetSeq()),
    ])
    input_li, output = pipe.fit_transform(df)

    # # modeling test
    # architecture = Architecture(pipe)
    # model = architecture.get_model()
    # model.fit(x=input_li, y=output, epochs=1)

    # evaluation test
    phone = "Samsung Galaxy S21 Ultra"
    model = tf.keras.models.load_model("/home/sh-sungho.park/my_git/slsi_df/02_WIP/21_WIP_SH/ModelFinalized/model.h5")
    evaluate = Evaluate(phone, df, df_init, pipe, model)

    evaluate.predict()

        

