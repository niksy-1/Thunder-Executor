import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

_nAgents = 10000
_nPatientZero = 5
_randomContacts = 30
_chanceOfInfection = 0.01
_chanceOfRecovery = 0.01

state = np.zeros(_nAgents)
state[:_nPatientZero] = 1

def infect(df, contacts):
    unique, counts = np.unique(contacts, return_counts=True)
    probability = 1 - np.power(1 - _chanceOfInfection, counts)
    change = np.random.uniform(0, 1, len(unique)) <= probability
    df.loc[unique, "state"] += change * np.maximum(1 - df.loc[unique, "state"], 0)

def recover(df):
    infected_indices = np.where(df["state"] == 1)[0]
    n_infected = len(infected_indices)
    n_recovered = int(_chanceOfRecovery * n_infected)
    if n_recovered > 0:
        recovered_indices = np.random.choice(infected_indices, size=n_recovered, replace=False)
        df.loc[recovered_indices, "state"] = 2

def step(df, nMarketplaces):
    nInfected = df["state"].sum()
    contacts = np.random.choice(df.index, int(_randomContacts * nInfected * nMarketplaces), replace=True)
    infect(df, contacts)
    recover(df)

def simulate_for_marketplaces(n_marketplaces):
    elapsed_periods_list = []
    for run in range(10):  # 10 runs for each marketplace number
        df = pd.DataFrame({"state": state.copy()})
        elapsed_periods = 0
        for _ in range(300):
            step(df, n_marketplaces)
            num_susceptible = (df["state"] == 0).sum()
            if num_susceptible == 0:
                break
            elapsed_periods += 1
        elapsed_periods_list.append(elapsed_periods)
    return np.mean(elapsed_periods_list)  # Return the average of the runs

# Utilize parallel processing to speed up the simulations
if __name__ == '__main__':
    with Pool() as p:
        marketplaces_range = list(range(1, 11))
        results = p.map(simulate_for_marketplaces, marketplaces_range)

    # Plotting
    plt.plot(marketplaces_range, results, marker='o', linestyle='-')
    plt.xlabel("Number of Marketplaces")
    plt.ylabel("Average Time Periods to Saturate Virus")
    plt.title("Effect of Marketplaces on Virus Saturation Time")
    plt.grid(True)
    plt.show()
