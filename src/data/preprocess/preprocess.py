import pandas as pd
from preprocess_utils import (convert_json_to_df, 
                             convert_date_format,
                             convert_to_null_value, 
                             convert_to_discrete_col)

class PreprocessAccounts:
    def __init__(self, df,
                 id_column, 
                 date_column, 
                 gender_column,
                 age_column,
                 address_column,
                 age_cut_list):
        self.df = df
        self.id_column = id_column
        self.date_column = date_column
        self.gender_column = gender_column
        self.age_column = age_column
        self.address_column = address_column

        self.age_cut_list = age_cut_list
        
    def preprocess(self):
        df = self.df

        df = convert_date_format(df, self.date_column, "seconds_to_datetime")
        df.dropna(inplace=True)

        gender_conditions = [
            lambda x: x == "-",
            lambda x: x == "0"
        ]
        df = convert_to_null_value(df, self.gender_column, gender_conditions)
        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=[self.gender_column])

        age_conditions = [
            lambda x: x == -1,
            lambda x: x <= 7
        ]
        df = convert_to_null_value(df, self.age_column, age_conditions)
        df.dropna(inplace=True)
        df = convert_to_discrete_col(df, self.age_column, "cut",
                                     cut_list=self.age_cut_list)
        df = pd.get_dummies(df, columns=[self.age_column])

        address_condition = lambda x: 0 if '경기' in x.split(' ')[0] else 1 \
                            if '인천' in x.split(' ')[0] else 1 \
                            if '서울' in x.split(' ')[0] else 1
        df[self.address_column] = df[self.address_column].apply(address_condition)
        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=[self.address_column])

        gen_col = [col for col in df.columns if col.startswith(self.gender_column)]
        add_col = [col for col in df.columns if col.startswith(self.address_column)]
        age_col = [col for col in df.columns if col.startswith(self.age_column)]
        
        df = df[[self.id_column, self.date_column] + gen_col + add_col + age_col]
        
        self.df = df
    
    def save(self, path, file_name):
        self.df.to_csv(path + file_name, encoding="utf-8", index=None)

class PreprocessProducts:
    def __init__(self, df,
                 pre_date_column,
                 post_date_column,
                 category_column,
                 price_column,
                 default_date,
                 days_list):
        self.df = df
        self.pre_date_column = pre_date_column
        self.post_date_column = post_date_column
        self.category_column = category_column
        self.price_column = price_column
        self.default_date = default_date
        self.days_list = days_list

    def preprocess(self):
        df = self.df

        df = df[df[self.pre_date_column].notna()]
        df = convert_date_format(df, self.pre_date_column, "custom_publish_date")
        df = convert_to_discrete_col(df, self.pre_date_column, "date_range",
                                     default_date=self.default_date,
                                     days_list=self.days_list,
                                     date_column=self.post_date_column)
        df = pd.get_dummies(df, columns=[self.post_date_column])

        df[self.category_column] = df[self.category_column].astype(str)
        df[self.category_column] = df[self.category_column].apply(lambda x: x[2:4] if len(x) > 3 else (x+((4-len(x))*'0'))[2:4])
        df = df[df[self.category_column] != "00"]
        df = pd.get_dummies(df, columns=[self.category_column])

        condition_1 = df[self.price_column] > 3500000.0 # 3500000 만원 이상은 고서적, 일반인들이 구매하지 않을 종류의 책들, 300000~3500000 가격의 책들은 대부분 전집에 해당
        condition_2 = df[self.price_column] < 1000.0 # 1000 미만은 사은품 등의 다른 제품들이 혼재
        df = df[~condition_1 & ~condition_2]
        df = df[df[self.price_column] != 0.0]
        df = convert_to_discrete_col(df, self.price_column, "qcut",
                                     cut_num=4)
        df = pd.get_dummies(df, columns=[self.price_column])

        self.df = df

    def save(self, path, file_name):
        self.df.to_csv(path + file_name, encoding="utf-8", index=None)

class PreprocessClicks:
    def __init__(self, df, device_column):
        self.df = df
        self.device_column = device_column

    def preprocess(self):
        df = self.df

        df = pd.get_dummies(df, columns=[self.device_column])
        df.dropna(inplace=True)

        self.df = df

    def save(self, path, file_name):
        self.df.to_csv(path + file_name, encoding="utf-8", index=None)

class PreprocessOrders:
    def __init__(self, df, 
                 date_column,
                 account_id_column,
                 product_id_column,
                 account_list,
                 product_list):
        self.df = df
        self.date_column = date_column
        self.account_column = account_id_column
        self.product_column = product_id_column
        self.account_list = account_list
        self.product_list = product_list

    def preprocess(self):
        df = self.df

        df = convert_date_format(df, self.date_column, "from_javatime")
        df = df[df[self.account_column].isin(self.account_list)]
        df = df[df[self.product_column].isin(self.product_list)]

        self.df = df

    def save(self, path, file_name):
        self.df.to_csv(path + file_name, encoding="utf-8", index=None)


if __name__ == "__main__":
    dirs = {"account" : ACCOUNT_DIR,
            "product" : PRODUCT_DIR,
            "click" : CLICK_DIR,
            "order" : ORDER_DIR}

    dfs = {} 
    for k, v in dirs.items():
        df = convert_json_to_df(v)
        dfs[k] = v

    accounts = dfs["account"]
    products = dfs["product"]
    clicks = dfs["click"]
    orders = dfs["order"]

    pa = PreprocessAccounts(accounts, 
                            ACCOUNT_ID_COLUMN,
                            ACCOUNT_DATE_COLUMN,
                            ACCOUNT_GENDER_COLUMN,
                            ACCOUNT_AGE_COLUMN,
                            ACCOUNT_ADDRESS_COLUMN,
                            AGE_CUT_LIST)
    pa.preprocess()
    pa.save(ACCOUNT_DIR, ACCOUNT_FILE_NAME)
    
    pp = PreprocessProducts(products, 
                            PRODUCT_PRE_DATE_COLUMN,
                            PRODUCT_POST_DATE_COLUMN,
                            PRODUCT_CATEGORY_COLUMN,
                            PRODUCT_PRICE_COLUMN,
                            PRODUCT_DEFAULT_DATE,
                            PRODUCT_DAYS_LIST)
    pp.preprocess()
    pp.save(PRODUCT_DIR, PRODUCT_FILE_NAME)

    pc = PreprocessClicks(clicks, CLICK_DEVICE_COLUMN)
    pc.preprocess()
    pc.save(CLICK_DIR, CLICK_FILE_NAME)

    account_list = pc.df[ACCOUNT_ID_COLUMN].unique().tolist()
    product_list = pc.df[PRODUCT_ID_COLUMN].unique().tolist()

    po = PreprocessOrders(orders, 
                          ORDER_DATE_COLUMN,
                          ACCOUNT_ID_COLUMN,
                          PRODUCT_ID_COLUMN,
                          account_list,
                          product_list)
    po.preprocess()
    po.save(ORDER_DIR, ORDER_FILE_NAME)