BILSTM_PARAMS = {
    'name': 'simple',
    'hidden_dim': 256,      # 256
    'dropout': 0.2,
    'n_layers': 2,
    'adv_number_of_layers': 2,
    'adv_dropout' : 0.2,
    'num_filters': 100,
    'filter_sizes': [3,4,5]
}


BILSTM_PARAMS_SIMPLER = {
    'name': 'simpler',
    'hidden_dim': 64,      # 256
    'dropout': 0.2,
    'n_layers': 2,
    'adv_number_of_layers': 2,
    'adv_dropout' : 0.2,
    'num_filters': 100,
    'filter_sizes': [3,4,5]
}



BILSTM_PARAMS_CONFIG3 = {
    'name' : 'three_layer',
    'hidden_dim': 512,
    'dropout': 0.5,
    'n_layers': 1,
    'adv_number_of_layers': 2,
    'adv_dropout': 0.4,
    'num_filters': 200,
    'filter_sizes': [3, 4, 5, 6]
}

BILSTM_PARAMS_CONFIG4 = {
    'name' : 'four_layer',
    'hidden_dim': 512,
    'dropout': 0.5,
    'n_layers': 2,
    'adv_number_of_layers': 2,
    'adv_dropout': 0.4,
    'num_filters': 200,
    'filter_sizes': [3, 4, 5, 6]
}

BILSTM_PARAMS_CONFIG5 = {
    'name' : 'five_layer',
    'hidden_dim': 256,
    'dropout': 0.5,
    'n_layers': 1,
    'adv_number_of_layers': 2,
    'adv_dropout': 0.4,
    'num_filters': 200,
    'filter_sizes': [3, 4, 5, 6]
}