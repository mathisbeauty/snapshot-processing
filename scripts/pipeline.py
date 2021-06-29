#!/usr/bin/env python
# coding: utf-8

# # SingularityDAO snapshot processing pipeline
# 
# This notebook it's meant to explore how the snapshots could be processed automatically.
# 
# [//]: <> (The code-only version can be found at `./scripts/pipeline.py`.)

# ## Parameters

# In[1]:


# Decimals AGI/AGIX

DECIMALS_AGI = 8

CONVERT_TO_FULL_BALANCE_AGI = 10 ** DECIMALS_AGI

AGI_THRESHOLD = 1000
AGI_THRESHOLD *= CONVERT_TO_FULL_BALANCE_AGI

# Decimals SDAO

DECIMALS_SDAO = 18

CONVERT_TO_FULL_BALANCE_SDAO = 10 ** DECIMALS_SDAO

# Reward parameters

TOTAL_STAKING_REWARD = 550000

TOTAL_REWARD = 825000

# Adjust rewards to be full balance

TOTAL_STAKING_REWARD *= CONVERT_TO_FULL_BALANCE_SDAO

TOTAL_REWARD *= CONVERT_TO_FULL_BALANCE_SDAO


# ## 1. Take snapshots
# 
# For the example pipeline, I'm going to use just a small number of snapshots taken manually, due to the fact that the main focus of this notebook is to create the processing pipeline, not to gather the snapshots.

# The first step is to import all the libraries that we're going to use for the data processing and for gathering insights about the dataset with statistical analysis.

# In[2]:


# Import libraries
# Pandas for tabular data
import pandas as pd
import numpy as np
from os import walk
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks

tqdm.pandas()


# Next step is reading the `data` directory to see how many snapshots we have for each token respectively (AGI and AGIX)

# In[3]:


def get_snapshots(path):
    return next(walk(path), (None, None, []))[2]

agi_snapshots_files = get_snapshots('../data/holders/agi')
agix_snapshots_files = get_snapshots('../data/holders/agix')

print(agi_snapshots_files)
print(agix_snapshots_files)


# Load snapshots for liquidity providers and stakers.
# 
# In this example, the snapshots are the same as the AGIX holders, to focus on developing the calculations first, leaving getting the data from the database for later on.

# In[4]:


lp_snapshots_files = get_snapshots('../data/lp')

stakers_snapshots_files = get_snapshots('../data/stakers')


# Read the snapshots' contents using `pandas` and convert balances to unsinged long numbers (SDAO wei)

# In[5]:


def read_csv(folder, file):
    # Read csv file
    data_frame = pd.read_csv('../data/%s/%s' % (folder, file))
    # Sort accounts by holding amount, larger holders at the top
    data_frame = data_frame.astype({'Balance': int })
    data_frame['Balance'] = data_frame['Balance'].apply(lambda x: int(x * CONVERT_TO_FULL_BALANCE_AGI))
    data_frame = data_frame.sort_values('Balance', ascending=False)
    return data_frame

agi_snapshots_raw = [read_csv("holders", "agi/" + file) for file in agi_snapshots_files]
agix_snapshots_raw = [read_csv("holders", "agix/" + file) for file in agix_snapshots_files]

lp_snapshots_raw = [read_csv("lp", file) for file in lp_snapshots_files]
stakers_snapshots_raw = [read_csv("stakers", file) for file in stakers_snapshots_files]


# The snapshots are now loaded as a panda DataFrame.
# 
# Let's see the structure of a single snapshot and a single row, to get a better idea of the dataset.

# In[6]:


print(agi_snapshots_raw[0].columns)
print(agi_snapshots_raw[0].iloc[0])


# Let's remove the `PendingBalanceUpdate` column and rename the other two, to clean the dataset and make it more practical.

# In[7]:


def clear_snapshot(snapshot):
    cleaned_snapshot = snapshot.drop('PendingBalanceUpdate', axis="columns")
    cleaned_snapshot = cleaned_snapshot.rename(columns={"HolderAddress": "address", "Balance": "balance"})
    cleaned_snapshot = cleaned_snapshot.reset_index(drop=True)
    return cleaned_snapshot

agi_snapshots = [clear_snapshot(snapshot) for snapshot in agi_snapshots_raw]
agix_snapshots = [clear_snapshot(snapshot) for snapshot in agix_snapshots_raw]

lp_snapshots = [clear_snapshot(snapshot) for snapshot in lp_snapshots_raw]
stakers_snapshots = [clear_snapshot(snapshot) for snapshot in stakers_snapshots_raw]

print(agi_snapshots[0].columns)
# Address and balance from the account with the largest holding, one of the Binance wallets
print(agi_snapshots[0].iloc[0])


# ## 2. Calculate eligibility
# 
# With the snapshot data ready, we can start to calculate the eligible addresses.
# 
# Note that the following is intended to illustrate the process, but with the actual dataset of snapshots, a similar but more complex processing would be applied.
# 
# Additionally, for the sake of the example, this process will only take into account holders, ignoring stakers and liquidity providers.
# 
# Nonetheless, it's feasible to adapt this same series of steps to stakers and liquidity providers, by preparing their respective datasets as described in step 1.

# ### Initial snapshot
# 
# There's an initial snapshot that delimits how many addresses are eligible for the airdrop.
# 
# In my case it's the snapshot of the frozen AGI balances, but in the airdrop it would be the snapshot from 17th of April 2021, at 23:59 UTC+0.
# 
# Let's create a subset based on the addresses from the first snapshot that have more than 1.000 AGI.

# In[8]:


print("AGI Snapshots: %s" % len(agi_snapshots))
print("AGIX Snapshots: %s" % len(agix_snapshots))
print("LP Snapshots: %s" % len(lp_snapshots))
print("Stakers Snapshots: %s" % len(stakers_snapshots))
print()

# Get the first snapshot and use it as the starting point for the calculations
def get_initial(initial_snapshot, category):
    total_addresses = len(initial_snapshot.index)
    eligible_addresses_initial = initial_snapshot[initial_snapshot['balance'] >= AGI_THRESHOLD]
    
    print('Total Addresses (%s): %s' % (category, total_addresses))
    print('Eligible Addresses (%s): %s' % (category, len(eligible_addresses_initial.index)))
    print()
    
    return eligible_addresses_initial

eligible_addresses_holders = get_initial(agi_snapshots[0], 'holders')
eligible_addresses_lp = get_initial(lp_snapshots[0], 'LP')
eligible_addresses_stakers = get_initial(stakers_snapshots[0], 'stakers')

print()
# Print address with smaller eligible balance
print(eligible_addresses_holders.iloc[-1])


# We can see that from the initial ~26k addresses, only 16689 pass the threshold to be eligible.
# 
# Now, it's a matter of iterating through the remaining snapshots using this initial set of accounts, and checking if the accounts are still eligible, removing the ones that are below the threshold.

# ### Iterate through the snapshots
# 
# First, let's merge all snapshots (AGI and AGIX) into a single array and discard the first one, as that one it's already processed.

# In[9]:


# Merge snapshots
holders_snapshots = agi_snapshots + agix_snapshots


# Now, we iterate over the snapshots, filtering the initial set of eligible accounts.

# In[10]:


def filter_addresses(initial_df, snapshot_df):
    # Calculate intersection of eligible addresses between existing set and snapshot set
    initial_set = set(initial_df['address'])
    snapshot_set = set(snapshot_df['address'])
    addresses_intersection = list(initial_set.intersection(snapshot_set))
    
    # Filter addresses based on whether they're contained on the intersection set or not
    filtered_df = initial_df[initial_df.apply(lambda x: x['address'] in addresses_intersection, axis=1)].copy()
    
    def filter_lowest_balance(x):
        return np.amin([x['balance'], snapshot_df.loc[snapshot_df['address'] == x['address']].iloc[0]['balance']])
    
    # Set balance amount to the lowest of the two values (initial value and snapshot value),
    # to only take into account the lower balance
    filtered_df['balance'] = filtered_df.copy().progress_apply(filter_lowest_balance, axis=1)
    
    return filtered_df

def get_eligible(initial_df, snapshots, category):
    print()
    print('Initial Eligible Addresses (%s): %s' % (category, len(initial_df.index)))
    print()

    eligible_df = initial_df

    for index, snapshot in enumerate(snapshots):
        print('Snapshot #%s' % index)
        snapshot_eligible = snapshot[snapshot['balance'] >= AGI_THRESHOLD]
        print('Eligible Addresses from snapshot: %s addresses' % len(snapshot_eligible.index))
        eligible_df = filter_addresses(eligible_df, snapshot_eligible)
        print('Eligible Addresses: %s' % len(eligible_df.index))
        print()

    print('Total Eligible Addresses (%s): %s' % (category, len(eligible_df.index)))
    
    return eligible_df
    

eligible_addresses_holders = get_eligible(eligible_addresses_holders, holders_snapshots, 'holders')
eligible_addresses_lp = get_eligible(eligible_addresses_lp, lp_snapshots, 'LP')
eligible_addresses_stakers = get_eligible(eligible_addresses_stakers, stakers_snapshots, 'stakers')


# ### Calculating airdrop amount

# #### Merge address lists
# 
# A single address could have multiple rewards. To account for this, we'll merge all the eligible addresses into a single list, removing duplicates, and add an extra column for each kind of reward.
# 
# That way, we'll be able to show all the reward types in the airdrop portal.

# In[11]:


datasets_dict = {
    "holder": eligible_addresses_holders,
    "lp": eligible_addresses_lp,
    "staker": eligible_addresses_stakers
}

# Merge all eligible addresses

addresses_df = pd.concat(list(datasets_dict.values())).drop_duplicates('address').drop('balance', axis=1)
addresses_df = addresses_df.reset_index(drop=True)

print('Total addresses participating in the airdrop: %s' % len(addresses_df.index))

# Append rewards to the provided column, matching addresses between both sets
def append_column_by_address(addresses_df, rewards_df, column, new_column_name=None):
    def get_row_value_by_address(address):
        matching_rows = rewards_df.loc[rewards_df['address'] == address]
        total_matching_rows = len(matching_rows)
        if total_matching_rows == 1:
            return matching_rows.iloc[0][column]
        elif total_matching_rows == 0:
            return 0
        else:
            raise Exception('Error appending column to final file', 'addresses are duplicated')
    
    result_df = addresses_df.copy()
    if new_column_name is None:
        new_column_name = column
    print()
    print('Appending "%s" column to final file' % new_column_name)
    result_df.insert(len(result_df.columns), new_column_name, addresses_df.progress_apply(lambda x: get_row_value_by_address(x['address']), axis=1).astype(np.longdouble))
    print()
    return result_df


# #### Stakers
# 
# There are two kinds of rewards for stakers:
# - Per user (divided equally among staking wallets)
# - Per stake amount (delivered proportionally to the amounts staked)

# In[12]:


print('Total Eligible Stakers: %s' % len(eligible_addresses_stakers.index))

rewards_stakers_df = eligible_addresses_stakers.copy()

# Rewards per user

half_staking_reward = TOTAL_STAKING_REWARD / 2

reward_per_user = half_staking_reward / len(eligible_addresses_stakers)

adjusted_reward_per_user = reward_per_user / CONVERT_TO_FULL_BALANCE_SDAO

print('Staking reward per user: %s' % adjusted_reward_per_user)

rewards_stakers_df.insert(len(rewards_stakers_df.columns), 'staker_reward_per_user', adjusted_reward_per_user)

# Rewards per stake

total_stake = eligible_addresses_stakers['balance'].sum()

rewards_stakers_df['staker_reward_per_stake'] = rewards_stakers_df.apply(lambda x: half_staking_reward * np.double(x['balance']) / np.double(total_stake) / CONVERT_TO_FULL_BALANCE_SDAO, axis=1)

print(rewards_stakers_df)

adjusted_half_staker_reward = (half_staking_reward / CONVERT_TO_FULL_BALANCE_SDAO)

calculated_staker_reward_per_user = np.sum(list(rewards_stakers_df['staker_reward_per_user']))

calculated_staker_reward_per_stake = np.sum(list(rewards_stakers_df['staker_reward_per_stake']))

print()
print('Allocated reward (stakers, per user): %s' % adjusted_half_staker_reward)
print('Calculated reward (stakers, per user): %s' % calculated_staker_reward_per_user)
print()
print('Allocated reward (stakers, per stake): %s' % adjusted_half_staker_reward)
print('Calculated reward (stakers, per stake): %s' % calculated_staker_reward_per_stake)
print()
print('Allocated reward (stakers, total): %s' % (TOTAL_STAKING_REWARD / CONVERT_TO_FULL_BALANCE_SDAO))
print('Calculated reward (stakers, total): %s' % (calculated_staker_reward_per_user + calculated_staker_reward_per_stake))
print()

per_user_error = calculated_staker_reward_per_user != adjusted_half_staker_reward

per_stake_error = calculated_staker_reward_per_stake != adjusted_half_staker_reward

total_error = (calculated_staker_reward_per_user + calculated_staker_reward_per_stake) != (adjusted_half_staker_reward * 2)

if per_user_error or per_stake_error or total_error:
    raise Exception('Error calculating rewards (stakers)', 'final reward sum does not match allocated reward')


# ##### Add the calculated rewards to final data frame

# In[13]:


rewards_stakers_df['balance'] /= CONVERT_TO_FULL_BALANCE_AGI

addresses_df = append_column_by_address(addresses_df, rewards_stakers_df, 'staker_reward_per_user')
addresses_df = append_column_by_address(addresses_df, rewards_stakers_df, 'staker_reward_per_stake')
addresses_df['staker_reward'] = addresses_df['staker_reward_per_user'] + addresses_df['staker_reward_per_stake']
addresses_df = append_column_by_address(addresses_df, rewards_stakers_df, 'balance', 'used_staker_balance')

print(addresses_df)


# #### Holders and LP
# 
# Knowing the eligibility of the addresses, we can calculate the balances now using the following formula.
# 
# With those premises in place, we can calculate the final reward for each user by using the following formula
# 
# `Reward = total_reward * log10(1+user_balance) / SUM(log10(1+user_balance))`

# In[14]:


# Define SUM(log10(1+user_balance)) as a constant variable

holder_balances = list(eligible_addresses_holders['balance'])

lp_balances = list(eligible_addresses_lp['balance'])

balances_log10 = [np.log10(1 + (balance)) for balance in (holder_balances + lp_balances)]

sum_balances_log10 = np.sum(balances_log10)

# Define the function that calculates the reward for each user

def calculate_reward(total_reward, user_balance_index):
    user_balance_log10 = balances_log10[user_balance_index]
    reward_percentage = np.longdouble(user_balance_log10) / np.longdouble(sum_balances_log10)
    # Calculate reward and convert to final balance
    return (total_reward * user_balance_log10 / sum_balances_log10) / CONVERT_TO_FULL_BALANCE_SDAO

# Calculate rewards and add the SDAO value as a column to the DateFrame

holder_rewards = [calculate_reward(TOTAL_REWARD, index) for index, balance in enumerate(holder_balances)]

lp_rewards = [calculate_reward(TOTAL_REWARD, len(holder_balances) + index) for index, balance in enumerate(lp_balances)]

holder_rewards_df = eligible_addresses_holders.copy()

lp_rewards_df = eligible_addresses_lp.copy()

holder_rewards_df.insert(0, 'reward', holder_rewards)

holder_rewards_df['balance'] /= CONVERT_TO_FULL_BALANCE_AGI

lp_rewards_df.insert(0, 'reward', lp_rewards)

lp_rewards_df['balance'] /= CONVERT_TO_FULL_BALANCE_AGI


# ##### Verify rewards

# In[15]:


# Verify that the total amount of allocated reward matches the expected value

calculated_reward = np.sum(list(holder_rewards_df['reward'])) + np.sum(list(lp_rewards_df['reward']))

adjusted_total_reward = (TOTAL_REWARD / CONVERT_TO_FULL_BALANCE_SDAO)

print('Allocated reward (holders and LP): %s' % adjusted_total_reward)
print('Calculated reward (holders and LP): %s' % calculated_reward)

if calculated_reward != adjusted_total_reward:
    raise Exception('Error calculating rewards', 'final reward sum does not match allocated reward')


# ##### Add the calculated rewards to final data frame

# In[ ]:


# Add rewards to final data frame

addresses_df = append_column_by_address(addresses_df, holder_rewards_df, 'reward', 'holder_reward')
addresses_df = append_column_by_address(addresses_df, holder_rewards_df, 'balance', 'used_holder_balance')
addresses_df = append_column_by_address(addresses_df, lp_rewards_df, 'reward', 'lp_reward')
addresses_df = append_column_by_address(addresses_df, lp_rewards_df, 'balance', 'used_lp_balance')
addresses_df['total_reward'] = addresses_df['staker_reward_per_user'] + addresses_df['staker_reward_per_stake'] + addresses_df['holder_reward'] + addresses_df['lp_reward']

total_calculated_reward = (addresses_df['total_reward'] * CONVERT_TO_FULL_BALANCE_SDAO).sum()

if total_calculated_reward != float(TOTAL_REWARD + TOTAL_STAKING_REWARD):
    print('Total rounding error: %s SDAO' % '{:.18f}'.format((TOTAL_REWARD + TOTAL_STAKING_REWARD) - (addresses_df['total_reward'] * CONVERT_TO_FULL_BALANCE_SDAO).sum()))
    raise Exception('Error calculating rewards', 'final reward sum does not match allocated reward')

total_calculated_reward /= CONVERT_TO_FULL_BALANCE_SDAO

# Sort addresses by total reward (descending) and recalculate indexes

addresses_df = addresses_df.sort_values('total_reward', ascending=False)
addresses_df = addresses_df.reset_index(drop=True)

print()
print('Allocated reward (stakers, holders and LP): %s' % (float(TOTAL_REWARD + TOTAL_STAKING_REWARD) / CONVERT_TO_FULL_BALANCE_SDAO))
print('Calculated reward (stakers, holders and LP): %s' % total_calculated_reward)
print()

print()
print('Final rewards')
print(addresses_df)


# ### Adding unclaimed balances
# 
# The missing step would be to sum all the unclaimed amounts from previous airdrops, for this example that's not possible with the data at hand though.
