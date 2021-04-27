import os
import yaml


class BHConfig:

    def __init__(self, config_file='data/bhconfig.yml'):

        self.config_file = config_file
        self._read_config()

    def _read_config(self):

        if os.path.isfile(self.config_file):
            with open(self.config_file) as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.config = {}

    def _update_config(self):

        with open(self.config_file, 'w') as file:
            yaml.dump(self.config, file)

    def _add_params(self, params):

        self.config = {**self.config, **params}
        self._update_config()
