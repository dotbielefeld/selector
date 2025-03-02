# Selector: Ensemble-based Automated Algorithm Configuration

An algorithm configurator in Python using multiple configuration generators and models derived from state-of-the-art methods. You can find documentation at (https://selector-ac.readthedocs.io/en/latest/).

# Currently integrated state-of-the-art AC method functionalities from

- CPPL
- GGA
- SMAC

# Installation

You can use selector from the files the github repository or install it via 

```
pip install swig
pip install selector-ac
```

# Usage

You can use Selectors facade in Python like this:

```
from selector.run_ac import ac

if __name__ == "__main__":
    scen_files = {'paramfile': 'your_path_to/params.pcs',
                  'instance_file': 'your_path_to/problem_instances.txt',
                  'feature_file': 'your_path_to/instance_features.txt'}

    ac(scen_files, 'desktop', # use 'cluster' for slurm
       run_obj='runtime', overall_obj='PAR10', cutoff_time=300,
       seed=44, par=10, winners_per_tournament=1, tournament_size=2,
       number_tournaments=2, termination_criterion='total_runtime',
       monitor='tournament_level', initial_instance_set_size=5, set_size=256,
       generator_multiple=1, memory_limit=2048, check_path=False,
       log_folder='your_log_folder_path', wallclock_limit=3600,
       wrapper_mod_name='your_path_to.your_wrapper', deterministic=0, 
       wrapper_class_name='Your_Wrapper')
```

You can also call a Python script as exemplified in selector/main.py, or call selector/main.py and pass paths and arguments via command line.

Selector will run the AC process until the 'wallclock_limit' or 'total_tournament_number' is reached and save the results in 'log_folder'.
