INPUT_SIZE = 310

OUTPUT_SIZE = 3

SUBJECTS_NUMBER = 5

SUBJECTS_NAME = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4']

DATA_NUMBER_OF_SUBJECT = 3397

EPOCH = 400

BATCH_SIZE = 32

ALPHA = 1

BETA = 0.31

LEARNING_RATE = 1e-5

SWEEP_CONFIG = {
    'name': 'dann',
    'metric': {'name': 'mean_accuracy', 'goal': 'maximize'},
    'method': 'bayes',
    'parameters': {
        'alpha': {
            'values': [0.1, 0.3, 0.6, 0.9]
        },
        'beta': {
            'values': [0.1, 0.3, 0.6, 0.9]
        },
    }
}
