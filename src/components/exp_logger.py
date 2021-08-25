import os
import time
import csv

import yaml


# for merging dictionaries
def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


class ExperimentLogger:
    def __init__(self, env_name="default_env", exp_name=None):
        # TODO: add seed to this thing
        self.exp_name = exp_name
        self.env_name = env_name

        # Set up the environment directory if this is the first time running this env (rare)
        self.env_path = f"./results/{env_name}/"
        try:
            os.mkdir(self.env_path)
        except OSError as e:
            print("Enviroment used beforehand")

        # Try setting up experiment
        experiments_tried = 0
        while True:
            try:
                # Set up experiment name if not created yet
                if exp_name is None:
                    exp_num_str = f"{len(os.listdir(self.env_path)) + experiments_tried}".zfill(3)
                    self.exp_name = f"Experiment #{exp_num_str}"

                self.exp_path = f"{self.env_path}{self.exp_name}/"
                os.mkdir(self.exp_path)
                break
            except OSError as e:
                print("Experiment by the same name already exists!")
                print(e)
                print("Retrying to create Experiment with different name")
                experiments_tried += 1

        # list of learner path directories
        self.learners = {}

        # logging general runtime data, like parameters of the learners
        self.runtime_data = {
            "total parameters": {}
        }

    def log_runtime_data(self):
        fname = f"{self.config_path}runtime_data.yaml"
        prev_data = self.read_config_file(fname)
        combined_data = merge(prev_data, self.runtime_data)
        print("\n" * 10)
        print(prev_data)
        print(self.runtime_data)
        print(combined_data)
        print("\n" * 10)
        self.log_config_file(fname, combined_data)

    def log_config(self, config):
        # create config folder
        self.config_path = f"{self.exp_path}config/"
        try:
            os.mkdir(self.config_path)
        except OSError as e:
            pass

        # log config
        self.log_config_file(f"{self.config_path}global_config.yaml", config)

    def log_config_file(self, fname, data):
        if len(data):
            with open(fname, 'w') as file:
                documents = yaml.dump(data, file)

    def read_config_file(self, fname):
        try:
            with open(fname, "r") as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        except OSError as e:
            print("Error")
            print(e)
            return {}

    def learner_name_to_path(self, learner_name):
        return f"{self.exp_path}{learner_name}/"

    def add_learner(self, learner_name):
        learner_path = self.learner_name_to_path(learner_name)

        # try setting up directory for learner
        try:
            os.mkdir(learner_path)
        except OSError as e:
            print("Learner already exists! Overriding previous tests hasn't been implemented yet (Delete manually)")
            print(e)
            exit()

        # Now we want to set up 2 files for the learner:
        # 1) A CSV containing all the information about the learner
        # 2) A CSV containing all the information about the enviroment
        paths = {
            "learner": f"{learner_path}learner_data.csv",
            "env": f"{learner_path}env_data.csv",
        }
        self.learners[learner_name] = paths

    def save_learner_data(self, learner_name, learner_data):
        # Assume learner data is a dictionary contatining rows and columns that
        # describe the run of the episode
        # We also assume that a column by the name of t_env exists as well
        # TODO: make the episode data an object?

        # Create name for this CSV
        path = self.learners[learner_name]["learner"]

        # Append CSV data
        csv_columns = list(learner_data.keys())

        try:
            self.write_row(path, csv_columns, learner_data)
        except IOError:
            print("I/O error")

    def save_env_data(self, learner_name, env_data):
        # Assume env data is a dictionary contatining rows and columns that
        # describe the run of the episode
        # We also assume that a column by the name of t_env exists as well
        # TODO: make the episode data an object?

        # Create name for this CSV
        path = self.learners[learner_name]["env"]

        # Append CSV data
        temp_cols = list(env_data.keys())
        temp_cols.remove("t_env")
        csv_columns = ['t_env'] + temp_cols

        try:
            self.write_row(path, csv_columns, env_data)
        except IOError:
            print("I/O error")

    def write_row(self, path, cols, data):
        # See if headers exists - is this the first time writig data?
        if not os.path.exists(path):
            with open(path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=cols)
                writer.writeheader()

        # append the row of data
        with open(path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writerow(data)
