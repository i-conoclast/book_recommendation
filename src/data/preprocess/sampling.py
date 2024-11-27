import pandas as pd
from sklearn.model_selection import train_test_split

def sample_accounts_by_distribution(data, account_id_column, date_column, sample_size, random_state=None):
    """
    Samples the click stream data while preserving the distribution of click counts per account.

    Parameters:
        data (pd.DataFrame): The click stream data containing account ID and click timestamp.
        user_id_col (str): The column name representing account IDs.
        click_time_col (str): The column name representing click timestamps.
        sample_size (int): The number of accounts to sample.
        random_state (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: A sampled DataFrame preserving click count distribution.
    """
    # Count clicks per account
    account_click_counts = data.groupby(account_id_column)[date_column].count().reset_index()
    account_click_counts.columns = [account_id_column, 'click_count']
    
    # Assign bins to click counts to stratify sampling
    account_click_counts['click_bin'] = pd.qcut(account_click_counts['click_count'], q=10, duplicates='drop')
    
    # Stratified sampling based on click_bin
    sampled_accounts = train_test_split(
        account_click_counts,
        stratify=account_click_counts['click_bin'],
        test_size=sample_size / len(account_click_counts),
        random_state=random_state
    )[1][account_id_column].tolist()
    
    return sampled_accounts

if __name__ == "__main__":
    clicks = pd.read_csv(PRE_PATH + PRE_CLICKS_FILE)
    orders = pd.read_csv(PRE_PATH + PRE_ORDERS_FILE)
    accounts = pd.read_csv(FE_PATH + FE_ACCOUNTS_FILE)
    products = pd.read_csv(FE_PATH + FE_PRODUCTS_FILE)
    
    sampled_accounts_ids = sample_accounts_by_distribution(clicks,
                                                       ACCOUNT_ID_COLUMN,
                                                       CLICK_DATE_COLUMN,
                                                       SAMPLE_SIZE,
                                                       42)
    
    sample_clicks = clicks[clicks[ACCOUNT_ID_COLUMN].isin(sampled_accounts_ids)]
    sample_orders = orders[orders[ACCOUNT_ID_COLUMN].isin(sampled_accounts_ids)]
    sample_accounts = accounts[accounts[ACCOUNT_ID_COLUMN].isin(sampled_accounts_ids)]
    
    sampled_products_ids = set.union(set(sample_clicks[PRODUCT_ID_COLUMN]), set(sample_orders[PRODUCT_ID_COLUMN]))
    sample_products = products[products[PRODUCT_ID_COLUMN].isin(sampled_products_ids)]

    sample_clicks.to_csv(SAMPLE_PATH + SMPL_CLICKS_FILE, index=None)
    sample_orders.to_csv(SAMPLE_PATH + SMPL_ORDERS_FILE, index=None)
    sample_accounts.to_csv(SAMPLE_PATH + SMPL_ACCOUNTS_FILE, index=None)
    sample_products.to_csv(SAMPLE_PATH + SMPL_PRODUCTS_FILE, index=None)

    