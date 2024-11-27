import pandas as pd
import numpy as np

def merge(accounts, 
          products,
          clicks,
          orders,
          account_id_column,
          product_id_column,
          order_id_column,
          click_date_column,
          drop_columns):
    
    interactions = pd.merge(clicks, orders, on=[account_id_column, product_id_column], how='left')
    interactions['reward'] = -1
    interactions.loc[interactions[interactions[order_id_column].isna()].index, 'reward'] = 0
    interactions.loc[interactions[interactions[order_id_column].notna()].index, 'reward'] = 1
    interactions = interactions[[account_id_column, product_id_column, click_date_column, "reward"]]

    acc_interactions = pd.merge(interactions, accounts, how="left", on=account_id_column)
    acc_pro_interactions = pd.merge(acc_interactions, products, how="left", on=product_id_column)

    acc_pro_interactions.drop(columns=drop_columns, inplace=True)

    acc_pro_interactions.rename(lambda x:'most_pref' + '_' + x.split('_')[-2] if x.endswith('x') else x,  axis=1, inplace=True)
    acc_pro_interactions.rename(lambda x:x[:-2] if x.endswith('y') else x,  axis=1, inplace=True)
    
    acc_pro_interactions.sort_values(click_date_column, inplace=True)

    return acc_pro_interactions

def classification_to_bandit_problem(contexts, labels, num_actions=None):
  """Normalize contexts and encode deterministic rewards."""

  if num_actions is None:
    num_actions = np.max(labels) + 1
  num_contexts = contexts.shape[0]

  # Due to random subsampling in small problems, some features may be constant
  sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

  # Normalize features
  contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

  # One hot encode labels as rewards
  rewards = np.zeros((num_contexts, num_actions))
  rewards[np.arange(num_contexts), labels] = 1.0

  return contexts, rewards, (np.ones(num_contexts), labels)


def safe_std(values):
  """Remove zero std values for ones."""
  return np.array([val if val != 0.0 else 1.0 for val in values])


def remove_underrepresented_classes(features, labels, thresh=0.0005):
  """Removes classes when number of datapoints fraction is below a threshold."""

  total_count = labels.shape[0]
  unique, counts = np.unique(labels, return_counts=True)
  ratios = counts.astype('float') / total_count
  vals_and_ratios = dict(zip(unique, ratios))
  print('Unique classes and their ratio of total: %s' % vals_and_ratios)
  keep = [vals_and_ratios[v] >= thresh for v in labels]
  return features[keep], labels[np.array(keep)]

def post_process(data, 
                 account_id_column,
                 product_id_column,
                 click_date_column):
    
    data = data[data["reward"] == 1]

    contexts = data.drop([account_id_column, product_id_column, click_date_column, "reward"])
    labels = data["book_cluster"]

    contexts, labels = remove_underrepresented_classes(contexts, labels)
    contexts, rewards, (opt_rewards, opt_actions) = classification_to_bandit_problem(contexts, labels)

    return contexts, rewards
