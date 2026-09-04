""" Plotting functions, this file is not tested"""

import matplotlib.pyplot as plt

def plot_train_ai_model(model_params, history):
    """
    Plot for ai training success.

    Parameters
    ----------
    model_params : dict
        Dictionary of model parameters.
    history : tensorflow object
        History of AAN training.
    """

    epoch = range(1, model_params['epochs'] + 1)

    plt.figure()
    plt.plot(epoch, history.history['extinction_mae'], label='Extinction', color='blue')
    plt.plot(epoch, history.history['scattering_mae'], color='green', label='Scattering')
    plt.plot(epoch, history.history['asymmetry_mae'], color='#57B9FF', label='Asymmetry')
    plt.plot(epoch, history.history['val_extinction_mae'], label='Val. Extinction',
             alpha=0.5, color='blue')
    plt.plot(epoch, history.history['val_scattering_mae'], label='Val. Scattering',
             color='green', alpha=0.5)
    plt.plot(epoch, history.history['val_asymmetry_mae'], label='Val. Asymmetry',
             color='#57B9FF', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {model_params['name']} Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epoch, history.history['extinction_loss'], label='Extinction', color='blue')
    plt.plot(epoch, history.history['scattering_loss'], color='green', label='Scattering')
    plt.plot(epoch, history.history['asymmetry_loss'], color='#57B9FF', label='Asymmetry')
    plt.plot(epoch, history.history['val_extinction_loss'], label='Val. Extinction',
             alpha=0.5, color='blue')
    plt.plot(epoch, history.history['val_scattering_loss'], label='Val. Scattering',
             color='green', alpha=0.5)
    plt.plot(epoch, history.history['val_asymmetry_loss'], label='Val. Asymmetry',
             color='#57B9FF', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model {model_params['name']} Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
