import random

from CustomError import *
from Modeling import *
from Pipelines import *

class ModelFinalized():
    def __init__(self):
        # define params
        self.data_raw_path = "/home/working/01_Data/01_Raw/df_base_reg_221109.parquet"
        self.data_init_path = "/home/working/01_Data/01_Raw/df_init_result_221109.parquet"
        self.data_type = "raw"

        # gpu setting
        gpus = tf.config.list_physical_devices(device_type="GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # warning setting
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        # set random_state
        SEED = 1
        os.environ["PYTHONASHSEED"]= str(SEED)
        os.environ["TF_DETERMINISTIC_OPS"]= "1"
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
    
    def check_validity(self, df):
        # check null value
        try : assert df.isna().sum().sum() == 0
        except: raise DataHasNullException

        # check whether zero value exists in unit
        try: assert df[df["unit"] <= 0].shape[0] == 0
        except: raise UnitHasZeroException

        # duplication check
        try:
            check_dup = df.groupby(["mkt_name", "month_no", "region", "company"]).size()
            assert check_dup[check_dup>1].shape[0] == 0
        except: raise DuplicationExistsException(check_dup)
    
    def get_data(self):
        # df_raw
        df_raw = pd.read_parquet(self.data_raw_path).reset_index(drop=True) # read data
        df_raw = df_raw[df_raw["dataset"] == self.data_type] # filter data type (raw, iqr, t99 ...)
        self.check_validity(df_raw) # validity check

        # df_init
        df_init = pd.read_parquet(self.data_init_path).reset_index(drop=True)
        df_init = df_init[df_init["dataset"] == self.data_type]
        df_init = df_init.rename(columns={"sim_pred_unit":"unit"})
        df_init["order_month"] = 1
        return df_raw, df_init
    
    def train_test_split(self, df_raw, df_init):
        test_phone_li = df_init["std_phone_nm"].unique()
        train = df_raw[~df_raw["std_phone_nm"].isin(test_phone_li)]
        test = df_raw[df_raw["std_phone_nm"].isin(test_phone_li)]
        return train
    
    def main(self, phone):
        df_raw, df_init = self.get_data() # get raw data
        train = self.train_test_split(df_raw, df_init) # get train data (test phones excluded)

        # define pipeline
        pipe = Pipeline([
            ("FeatureEngineer", FeatureEngineer()),
            ("Encode", Encode()),
            ("GetSeq", GetSeq()),
        ])

        input_data_li, output_data = pipe.fit_transform(train) # get dataset

        # get model and fit
        architecture = Architecture(pipe)
        model = architecture.get_model()

        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-2, min_lr=1e-10, monitor="val_loss", factor=0.1)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1e-2, monitor="val_loss")
        model.fit(x=input_data_li, y=output_data, eopchs=1000, callbacks=[lr_schedule, early_stopping], validation_split=0.1)
        model.save("/home/sh-sungho.park/my_git/slsi_df/02_WIP_SH/ModelFinalized/model.h5")
        evaluate = Evaluate(phone, df_raw, df_init, pipe, model)
        df_result = evaluate.predict()
        return df_result

if __name__ == "__main__":
    mf = ModelFinalized()
    df_raw, df_init = mf.get_data()
    phone = df_init["std_phone_nm"].unique()[0]
    df_result = mf.main(phone)
    print(df_result)