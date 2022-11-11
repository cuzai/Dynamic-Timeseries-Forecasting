import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def make_month(self, _x):
        x = _x.copy()
        x["month"] = x["month_no"].str[-2:].astype(int)
    
    def make_cyclic_month(self, _x):
        x = _x.copy()
        x["month_norm"] = 2 * np.pi * x["month"] / 12
        x["cos_month"] = np.cos(x["month_norm"])
        x["sin_month"] = np.sin(x["month_norm"])
        x = x.drop("month_norm", axis=1)
        return x
    
    def transform(self, x, y=None):
        x = self.make_month(x)
        x = self.make_cyclic_month(x)
        return x

class Encode(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.region_encoder = LabelEncoder()
        self.company_encoder = LabelEncoder()
        self.month_encoder = LabelEncoder()

        self.region_encoder.fit(x["region"])
        self.company_encoder.fit(x["company"])
        self.month_encoder.fit(x["month"])
        return self
    
    def transform(self, _x, y=None):
        x = _x.copy()
        x["region"] = self.region_encoder.transform(x["region"])
        x["company"] = self.company_encoder.transform(x["company"])
        x["month"] = self.month_encoder.transform(x["month"])

        # make a room for padding zero
        x["region"] += 1
        x["company"] += 1
        x["month"] += 1

        return x
    
class GetSeq(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_train = True
    
    def get_maxlen(self, x):
        return x.groupby(["region", "company", "std_phone_nm"]).size().max()
    
    def fit(self, x, y=None):
        self.maxlen = self.get_maxlen(x)
        return self

    def func(self, x):
        if self.is_train: # dataset for training
            self.input_unit.append(x["unit"].values[:-1])
            self.input_sinmonth.append(x["sin_month"].values[:-1])
            self.input_cosmonth.append(x["cos_month"].values[:-1])

            self.input_region.append(x["region"].values[:-1])
            self.input_company.append(x["company"].values[:-1])
            self.input_month.append(x["month"].values[:-1])

            self.output.append(x["unit"].values)
        else: # dataset for test
            self.input_unit.append(x["unit"].values)
            self.input_sinmonth.append(x["sin_month"].values)
            self.input_cosmonth.append(x["cos_month"].values)

            self.input_region.append(x["region"].values)
            self.input_company.append(x["company"].values)
            self.input_month.append(x["month"].values)
    
    def transform(self, x, y=None):
        self.input_unit = []
        self.input_sinmonth = []
        self.input_cosmonth = []

        self.input_region = []
        self.input_company = []
        self.input_month = []

        self.output = []

        x.groupby(["region", "company", "std_phone_nm"]).apply(lambda p: self.func(p))

        self.input_unit = tf.keras.preprocessing.sequence.pad_sequnce(self.input_unit, value=0, maxlen=self.maxlen, dtype="float")
        self.input_sinmonth = tf.keras.preprocessing.sequence.pad_sequnce(self.input_sinmonth, value=0, maxlen=self.maxlen, dtype="float")
        self.input_cosmonth = tf.keras.preprocessing.sequence.pad_sequnce(self.input_cosmonth, value=0, maxlen=self.maxlen, dtype="float")

        self.input_region = tf.keras.preprocessing.sequence.pad_sequnce(self.input_region, value=0, maxlen=self.maxlen, dtype="float")
        self.input_company = tf.keras.preprocessing.sequence.pad_sequnce(self.input_company, value=0, maxlen=self.maxlen, dtype="float")
        self.input_month = tf.keras.preprocessing.sequence.pad_sequnce(self.input_month, value=0, maxlen=self.maxlen, dtype="float")

        self.output = tf.keras.preprocessing.sequence.pad_sequnce(self.output, value=0, maxlen=self.maxlen, dtype="float")
        self.output = self.output.reshape(-1, self.output.shape[1], 1)
        self.is_train = False

        return (self.input_unit, self.input_sinmonth, self.input_cosmonth, self.input_region, self.input_company, self.input_month), self.output

if __name__ == "__main__":
    df = pd.read_parquet("/home/working/01_Data/01_Raw/df_base_reg_221109.parquet")
    input_li, output = Pipeline([
        ("FeatureEngineer", FeatureEngineer()),
        ("Encode", Encode()),
        ("GetSeq", GetSeq()),
    ]).fit_transform(df)
    
    print(input_li[0].shape)
    print(output.shape)