# LOMAQ - Local Multi-Agent Q-Learning

This repository offers a scalable value decomposition method for the cooperative CTDE setting. 
Our method leverages local agent rewards for improving credit assignment, whilst maintaining a 
cooperative objective. In addition, we provide a direct decomposition method for finding local 
rewards when only a global reward is provided.

## Installation

The code is written in Python 3, with extensive use of the Pytorch library. Installation of the
relevant python packages can be done by running

```shell
cd LOMAQ/
pip install -r requirements.txt
```

## Configuring parameters

Our method is dependent on hyper-parameters that can be found under `src/config/`. There are 4 different 
types of configuration files

* **Default Configuration** - found under `src/default.yaml`. This file depicts the default parameters
for all runs. Is overriden by any other configuration file. An example of a parameter here could be
  `batch_size`, which is generally similar for all algorithms. 

* **Enviroment Configuration** - found under `src/envs/<env-name>.yaml`. This file depicts the parameters
that are relevant for running the enviroment. An example of this is `num_agents`. 
  
* **Algorithm Configuration** - found under `src/algs/<alg-name>.yaml`. This file depicts the parameters
for the current algorithm. An example of this is `learning_rate`. 
  
* **Test Configuration** - found under `src/test/<test-name>.yaml`. This file depicts information for
a specific test. For instance, if one wishes to run a certain algorithm against different num_agents, 
  this should be done using a seperate test file. Each test file depicts a series of runs. See 
  `src/test/example_test.yaml` for an example of this. This config file overrides all other configurations. 
  
## Running the Code

We offer a variety of different ways for running the code for different purposes. All the mentioned 
bash files can be found under `scripts/`. Note that running the code should always be done from the 
main `LOMAQ/` directory, and **NOT** from `src/` or `scripts/`.

### Simple Run

The simplest way to run the code is using `src/main.py`, which expects exactly 2 arguments - Am environment
name, and an algorithm name. The code assumes that these names correspond to 2 configuration files that depict
the relevant parameters. For instance:

```shell
python3 src/main.py --env-name=lomaq --alg-name=multi_particle
```

An alternative is running `scripts/run_bash.sh`. By default, the script will run `src/main.py` with LOMAQ and the 
multi_particle environment.

### Running a Test

If one wishes to run a test from `src/test/`, this can be done using `src/multi_main.py`. The python script will 
read the relevant test file, and initiate the runs that it depicts. The runs are run in parallel. For example, 
if one wishes to run test number `53`:

```shell
python3 src/multi_main.py --test-num=53 --platform=bash
```

If one wishes to run the same test multiple times with different seeds, this can be done using another argument 
`iteration_num` (default is 0). The corresponding seed will be `iteration_num + test_num`, and can be done using:

```shell
python3 src/multi_main.py --test-num=53 --iteration-num=10 --platform=bash
```

If one wishes to only run a single run from a test, this can be done using `src/single_main.py` by specifying the 
`run_num` argument:

```shell
python3 src/single_main.py --test-num=53 --iteration-num=10 --run-num=2
```