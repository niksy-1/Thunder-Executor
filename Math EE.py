import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

_nAgents = 10000
_nPatientZero = 5
_randomContacts = 30
_chanceOfInfection = 0.01
_chanceOfRecovery = 0.01
_nMarketplaces = 3  # Number of marketplaces

state = np.zeros(_nAgents)
state[:_nPatientZero] = 1
df = pd.DataFrame({"state": state})

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

def step(df):
    nInfected = df["state"].sum()
    contacts = np.random.choice(df.index, int(_randomContacts * nInfected), replace=True)
    infect(df, contacts)
    recover(df)

def plot_agents(df, elapsed_periods, num_marketplaces):
    # Create a 2D grid representing the agents
    grid_size = int(np.sqrt(_nAgents))
    grid = np.zeros((_nAgents // grid_size, grid_size), dtype=int)

    # Mapping of agent state to color
    color_map = {0: "green", 1: "red", 2: "blue"}

    # Create custom colormap
    cmap = plt.cm.colors.ListedColormap([color_map[0], color_map[1], color_map[2]])

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Agent State Visualization")
    ax.set_xticks([])
    ax.set_yticks([])

    # Create the initial plot
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Susceptible', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Infected', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Recovered', markerfacecolor='blue', markersize=10)
    ]

    # Add marketplace legend elements and text
    for i in range(num_marketplaces):
        color = plt.cm.colors.to_rgba(f"C{i}", alpha=0.5)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Marketplace {i+1}',
                                          markerfacecolor=color, markersize=10))
    ax.legend(handles=legend_elements, loc='upper right')

    # Create text annotation for elapsed periods and number of marketplaces
    text_ann = ax.annotate("Elapsed Periods: 0\nMarketplaces: 0", xy=(0.01, 0.95), xycoords='axes fraction', fontsize=10)

    plt.show(block=False)

    for _ in range(_nExperiments):
        df = pd.DataFrame({"state": state})

        for _ in tqdm(range(_nSteps), ncols=80):
            step(df)

            # Update the grid with the current agent states
            grid = np.reshape(df["state"].to_numpy(), grid.shape)

            # Update the plot with the new grid
            img.set_data(grid)

            # Update the text annotation with the current elapsed periods and number of marketplaces
            text_ann.set_text(f"Elapsed Periods: {elapsed_periods}\nMarketplaces: {num_marketplaces}")

            plt.pause(0.5)
            plt.draw()

            # Check the number of susceptible people
            num_susceptible = (df["state"] == 0).sum()
            if num_susceptible == 0:
                break

            elapsed_periods += 1  # Increment the counter for each time period

        # Check if the loop was broken
        if num_susceptible == 0:
            break

    plt.show()

    print("Total time periods elapsed:", elapsed_periods)


_nExperiments = 10
_nSteps = 300  # Increased time period duration

elapsed_periods = 0  # Counter for the number of time periods elapsed

plot_agents(df, elapsed_periods, _nMarketplaces)
