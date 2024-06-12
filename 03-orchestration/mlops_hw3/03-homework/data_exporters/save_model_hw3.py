import os
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(final_lr_model):
    """
    """
    # Specify your data exporting logic here
    cwd = os.getcwd()
    filename = f'{cwd}/finalized_model_hw03.lib'
    print(f'Saving model to {filename}')
    pickle.dump(final_lr_model, open(filename, 'wb'))    

