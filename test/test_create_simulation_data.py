# Library
import sys
import pytest
from pathlib import Path
import numpy as np

# Path 
try:
    PROJECT_PATH = Path(__file__).parents[1]
except NameError:
    PROJECT_PATH = Path('.').absolute().parents[1]
sys.path.insert(0, str(Path(PROJECT_PATH, 'code')))

# Test
NUM_CLIENT = 5
SAMPLE_PER_CLIENT = 10

@pytest.fixture()
def simulation_data():
    from create_simulation_data import SimulationData
    return SimulationData(SAMPLE_PER_CLIENT, NUM_CLIENT)

def test_edge_synthetic_data(simulation_data):
    x = simulation_data.x
    y1 = simulation_data.edge_synthetic_data[:,:,0]
    y2 = simulation_data.edge_synthetic_data[:,:,1]
    assert simulation_data.edge_synthetic_data.shape == (NUM_CLIENT, SAMPLE_PER_CLIENT, 2) 
    assert np.array_equal(np.sin(x), y1)
    assert np.array_equal(np.cos(x), y2)

def test_cloud_synthetic_data(simulation_data):
    x = simulation_data.x
    y1 = simulation_data.cloud_synthetic_data[:,:,0]
    y2 = simulation_data.cloud_synthetic_data[:,:,1]
    assert simulation_data.cloud_synthetic_data.shape == (NUM_CLIENT, SAMPLE_PER_CLIENT, 2) 
    assert np.array_equal(3 * x + 2, y1)
    assert np.array_equal((5 * x)**2 + 4 * x, y2)

def test_label(simulation_data):
    assert simulation_data.label_value.shape == (NUM_CLIENT, SAMPLE_PER_CLIENT)  
    