import os
import time
import csv


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

        # Set up experiment name if not created yet
        if exp_name is None:
            self.exp_name = f"Experiment #{len(os.listdir(self.env_path))}"

        # Try setting up experiment
        self.exp_path = f"{self.env_path}{self.exp_name}/"
        try:
            os.mkdir(self.exp_path)
        except OSError as e:
            print("Experiment by the same name already exists!")
            print(e)
            exit()

        # list of learner path directories
        self.learners = []

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

        self.learners.append(learner_path)

    def save_episode(self, learner_name, episode_data, episode_name=None):
        # Assume episode data is a dictionary contatining rows and columns that
        # describe the run of the episode
        # We also assume that a column by the name of t_env exists as well
        # TODO: make the episode data an object?

        # Create name for this CSV
        learner_path = self.learner_name_to_path(learner_name)
        if episode_name is None:
            episode_name = f"output_run{len(os.listdir(learner_path))}"
        episode_path = f"{learner_path}{episode_name}.csv"

        # Create CSV data
        temp_cols = list(episode_data[0].keys())
        temp_cols.remove("t_env")
        csv_columns = ['t_env'] + temp_cols
        try:
            with open(episode_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in episode_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

