class DataHasNullException(Exception):
    def __init__(self, df):
        self.df = df
    
    def __str__(self):
        return f"\n{self.df.isna().sum()}"
    
class UnitHasZeroException(Exception):
    def __init__(self, df):
        self.df = df
    
    def __str__(self):
        return f"\n{self.df[self.df['unit'] <=0]}"

class DuplicationExistsException(Exception):
    def __init__(self, df):
        self.df = df
    
    def __str__(self):
        return f"\n{self.df[self.df > 1]}"