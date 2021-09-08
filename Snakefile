rule raw_data:
    input:
    output:
        "data/raw/simulation_cloud_data.pkl",
        "data/raw/simulation_edge_data.pkl",
        "data/raw/simulation_label_data.pkl"
    script:
        "code/create_simulation_data.py"
    