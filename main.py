from utils.file import get_rc_training_data, get_rc_validation_data
from utils.nn import train

if __name__ == '__main__':
    print('Loading training dataset')
    training_data = get_rc_training_data()
    print('Loading validation dataset')
    validation_data = get_rc_validation_data()
    print('Loaded datasets')
    print('Starting training')
    train(training_data, validation_data)
