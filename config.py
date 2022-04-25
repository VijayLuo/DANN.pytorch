INPUT_SIZE = 310

OUTPUT_SIZE = 3

SUBJECTS_NUMBER = 5

SUBJECTS_NAME = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4']

DATA_NUMBER_OF_SUBJECT = 3397

EPOCH = 400

BATCH_SIZE = 16

ALPHA = 1

BETA = 0.31

LEARNING_RATE = 1e-5

SWEEP_CONFIG = {
    'name': 'dann',
    'metric': {'name': 'mean_accuracy', 'goal': 'maximize'},
    'method': 'bayes',
    'parameters': {
        'alpha': {
            'min': 0.1,
            'max': 1,
            'distribution': 'uniform',
        },
        'beta': {
            'min': 0.1,
            'max': 1,
            'distribution': 'uniform',
        },
    }
}
