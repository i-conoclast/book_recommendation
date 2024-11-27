import pandas as pd
from functools import reduce

class FeatureEngineering:
    def __init__(self, 
                 accounts,
                 products,
                 clicks,
                 orders,
                 account_id_column,
                 product_id_column,
                 order_id_column,
                 product_date_column,
                 order_date_column,
                 click_date_column,
                 pref_column):
        self.accounts = accounts
        self.products = products
        self.clicks = clicks
        self.orders = orders

        self.account_id_column = account_id_column
        self.product_id_column = product_id_column
        self.order_id_column = order_id_column
        self.product_date_column = product_date_column
        self.order_date_column = order_date_column
        self.click_date_column = click_date_column
        self.pref_column = pref_column

    def make_preference(self, mode):
        products = self.products

        # 2020년 이전에 출판된 도서는 모두 예전 도서로 취급
        products['pub_newold'] = products[self.product_date_column].astype(str).apply(lambda x: 1 if x[0:4] == '2020' else 0)
        products.dropna(inplace=True)

        if mode == "order_base":
            data = self.orders
            date_column = self.order_date_column
        if mode == "click_base":
            data = self.clicks
            date_column = self.click_date_column
        
        data[self.order_date_column] = data[date_column].astype("datetime64[ns]")
        data_products = pd.merge(data, products, on="product_id", how="left")
        data_products.dropna(inplace=True)

        preference = pd.DataFrame(data_products.groupby(self.account_id_column)['pub_newold'].agg('mean')).reset_index()
        preference.rename(columns = {'pub_newold':'new_preference'}, inplace=True)

        data_preference = pd.merge(data, preference, on="account_id", how="left")
        data_preference['new_pref'] = data_preference['new_preference'].apply(lambda x: 1 if x>=0.5 else 0)
        data_preference = data_preference.groupby('account_id').mean()['new_preference'].reset_index() # 확인 필요

        if mode == "order_base":
            self.orders_preference = data_preference 
        if mode == "click_base":
            self.clicks_preference = data_preference


    def make_activity_times(self):
        clicks = self.clicks
        
        clicks['time'] = clicks[self.click_date_column].str.slice(start=11, stop=13)
        condition = lambda x: 'day' if int(x) in list(range(6,19)) else 'night'
        clicks['click_time'] = clicks['time'].apply(condition)
        clicks['day'] = clicks[self.click_date_column].str.slice(start=0,stop=11)
        clicks['week_day'] = pd.to_datetime(clicks.day).dt.day_name()
        clicks['weekend'] = clicks['week_day'].apply(lambda x: 1 if x in ['Sunday','Saturday'] else 0)
        
        clicks = pd.get_dummies(clicks, columns=['click_time'])
        clicks = pd.get_dummies(clicks, columns=['weekend'])

        clicks = clicks.drop('week_day', axis=1)
        clicks = clicks.groupby('account_id').sum()

        clicks['day_ratio'] = clicks.apply(lambda x:x['click_time_day'] / (x['click_time_day'] + x['click_time_night']), axis=1)
        clicks['weekend_ratio'] = clicks.apply(lambda x:x['weekend_1'] / (x['weekend_0'] + x['weekend_1']), axis=1)

        clicks = clicks[["click_time_day", "click_time_night", "weekend_0", "weekend_1", "day_ratio", "weekend_ratio"]]
        activity_times = clicks.reset_index()
        self.activity_times = activity_times

    def make_category_preference(self):
        accounts = self.accounts
        products = self.products
        clicks = self.clicks

        clicks_products = pd.merge(clicks, products, on=self.product_id_column, how="left")
        cate_cols = [col for col in clicks_products.columns if col.startswith('category')]
        clicks_products = clicks_products[[self.account_id_column, self.product_id_column] + cate_cols]

        accounts_clicks_products = pd.merge(accounts, clicks_products, how="left", on=self.account_id_column)
        accounts_clicks_products = accounts_clicks_products.groupby(self.account_id_column).sum()[cate_cols]
        category_preference = accounts_clicks_products.div(accounts_clicks_products.sum(axis=1), axis="index").fillna(0)
        self.category_preference = category_preference.reset_index()

    def make_involvement(self):
        clicks = self.clicks
        orders = self.orders

        clicks = clicks.groupby(self.account_id_column).count().iloc[:, 0] # 확인 필요
        clicks = clicks.reset_index()

        orders = orders.groupby(self.account_id_column).count()[self.order_id_column]
        orders = orders.reset_index()

        involvement = clicks.div(orders, axis="index").reset_index().rename({0:'involvement'}, axis=1)
        self.involvement = involvement

    def make_bestseller(self):
        products = self.products
        orders = self.orders

        cate_cols = [col for col in products.columns if col.startswith('category')]
        products["category_id"] = products[cate_cols].idxmax(axis=1).str[-2:]
        products = products[["category_id", self.product_id_column]]

        bestseller = pd.merge(orders, products, on=self.product_id_column, how="left")
        bestseller.dropna(inplace=True)

        best_by_category = pd.DataFrame(bestseller.groupby(['category_id', self.product_id_column]) \
                                        .count()[self.order_date_column]) \
                                        .sort_values(self.order_date_column, ascending=False) # 왜 order_id로 하지 않았는지?
        best200_nocat = best_by_category.sort_values([self.order_date_column],ascending=False)[0:200]\
                        .reset_index()[self.product_id_column].tolist()
        
        cat_best24_list = []
        for i in range(0, 24):
            category_best24 = best_by_category.sort_values(['category_id', self.order_date_column], ascending=False)\
                                              .unstack(level=0).iloc[:, i]\
                                              .sort_values(ascending=False)[:24].index.to_list()
            cat_best24_list.extend(category_best24)

        products['best200'] = products[self.product_id_column].apply(lambda x:1 if x in best200_nocat else 0)
        products['bestcat24'] = products[self.product_id_column].apply(lambda x:1 if x in cat_best24_list else 0)

        self.bestseller = products[[self.product_id_column, "best200", "bestcat24"]]

    def merge(self):
        accounts = self.accounts
        accounts_merge_list = []

        merged_preference = self.orders_preference.copy()
        merged_preference["new_preference"] = (0.7 * self.orders_preference["new_preference"] + (0.3 * self.clicks_preference["new_preference"]))
        accounts_merge_list.append(merged_preference)
        accounts_merge_list.append(self.activity_times)
        accounts_merge_list.append(self.category_preference)
        accounts_merge_list.append(self.involvement)

        #account_merge = reduce(lambda left, right: pd.merge(left, right, on=self.account_id_column, how="left"), accounts_merge_list)
        for df in accounts_merge_list:
            accounts = pd.merge(accounts, df, how='left', on=self.account_id_column)
        self.account_merge = accounts

        products = self.products
        products = pd.merge(products, self.bestseller, how='left', on=self.product_id_column)
        self.product_merge = products

if __name__ == "__main__":
    pass