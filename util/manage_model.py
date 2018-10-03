import torch
import os


def load(model, device, parameter_dir=None, force_load=None):
    mainModel = model().to(device)
    optimizer = torch.optim.RMSprop(mainModel.parameters(), lr=2.5e-4)
    step = 1

    epoch_to_load = 0
    if parameter_dir is not None:
        for _, _, files in os.walk(parameter_dir):
            for parameter_file in files:
                # The name of parameter file is {epoch}.save
                name, extension = parameter_file.split('.')
                epoch = int(name)

                if epoch > epoch_to_load:
                    epoch_to_load = epoch

    if force_load is not None:
        epoch_to_load = force_load

    if epoch_to_load != 0:
        parameter_file = '{parameter_dir}/{epoch}.save'.format(parameter_dir=parameter_dir, epoch=epoch_to_load)
        parameter = torch.load(parameter_file)

        mainModel.load_state_dict(parameter['state'])

    return mainModel, optimizer, step, epoch_to_load
