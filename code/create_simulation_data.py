#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NUM_CLIENT = 100
SAMPLE_PER_CLIENT = 1000

try:
    PROJECT_PATH = Path(__file__).parents[1]
except NameError:
    PROJECT_PATH = Path('.').absolute().parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data', 'raw')

#%%
class SimulationData:
    def __init__(self, sample_per_client, num_client):
        self.sample_size = sample_per_client
        self.num_client = num_client
        self.x = np.random.normal(0, 1, (self.num_client, self.sample_size))
        self.e = np.random.normal(0, 0.1, (self.num_client, self.sample_size))
        # TODO add num of value, error term shape value --> 2 to n
        self.edge_synthetic_data = self.edge_synthetic()
        self.cloud_synthetic_data = self.cloud_synthetic()
        self.label_value = self.label()

    def edge_synthetic(self):
        return np.array([np.sin(self.x), 
                         np.cos(self.x)]) \
                .transpose(1, 2, 0)

    def cloud_synthetic(self):
        return np.array([3 * self.x + 2, 
                         (5 * self.x) ** 2 + 4 * self.x]) \
                .transpose(1, 2, 0)

    def label(self):
        return np.sum(self.edge_synthetic_data, axis=2) * np.sum(self.cloud_synthetic_data, axis=2) + self.e
#%%
if __name__ == "__main__":
    simulation_data = SimulationData(SAMPLE_PER_CLIENT, NUM_CLIENT)
    
    with open(Path(DATA_PATH, 'simulation_edge_data.pkl'), 'wb') as f:
        pickle.dump(simulation_data.edge_synthetic_data, f)
    
    with open(Path(DATA_PATH, 'simulation_cloud_data.pkl'), 'wb') as f:
        pickle.dump(simulation_data.cloud_synthetic_data, f)
    
    with open(Path(DATA_PATH, 'simulation_label_data.pkl'), 'wb') as f:
        pickle.dump(simulation_data.label_value, f)
    
    plt.scatter(simulation_data.x.flatten(), 
                simulation_data.cloud_synthetic_data[:,:,0].flatten())
    plt.scatter(simulation_data.x.flatten(), 
                3 * (simulation_data.x.flatten()) + 2)
    
    plt.scatter(simulation_data.x.flatten(), 
                simulation_data.cloud_synthetic_data[:,:,1].flatten())   
    plt.scatter(simulation_data.x.flatten(), 
                (5 * simulation_data.x.flatten()) ** 2 + 4 * (simulation_data.x.flatten()))

