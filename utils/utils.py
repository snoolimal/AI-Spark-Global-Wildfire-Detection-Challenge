def check_mode(mode):
    if mode not in ['train', 'test']:
        raise ValueError("Parameter 'mode' must be either 'train' or 'test'.")
    # assert mode in ['trani', 'test'], "Parameter 'mode' must be either 'train' or 'test'."
