.. _usage:

Usage
=====

You need to provide the following for an AC run with Selector:

- A .pcs file with a parameter space definition according to PCS (`PCS Manual <https://aclib.net/cssc2014/pcs-format.pdf>`_, `PCS Example <https://github.com/YashaPushak/PCS/blob/master/examples/params-cplex.pcs>`_) and the path to it.
- Instances: A problem instance set for training, a .txt with paths to the instances (one path per line) and the path to it.
- Instance features: A .txt file with the first line as headers, with INSTANCE_NAME,featureName1,featureName2..., and a line with the instance name and feature values for each instance in the training set and the path to it. If you cannot provide features, set up mock features, e.g. a few features with all values set to 1. However, providing meaningful features will increase the AC success.
- A parameterized target algorithm that can be called via shell and passed paraemeters to.
- A Python wrapper for the target algorithm, its name and the name of the class. It is of the following form:

.. code-block:: python

	import argparse
	import ast


	class Your_Wrapper():

	    def get_command_line_args(self, runargs, config):

	        instance = runargs["instance"]
	        id = runargs["id"]

	        configuration =""
	        for param, value in config.items():
	                configuration += f" --{param}={value}"

	        exc = '/absolute/path/to/your/target_alorithm'
	        # Example command
	        cmd = f"stdbuf -oL {exc} {configuration} -instance {instance}"
	        return cmd

	if __name__ == "__main__":
	    parser = argparse.ArgumentParser()

	    parser.add_argument('--runargs',type=ast.literal_eval)
	    parser.add_argument('--config',type=ast.literal_eval)
	    args = vars(parser.parse_args())

	    wrapper = Your_Wrapper()

- The value 'desktop' or 'cluster' set to ray_mode, if using the Selector facade (see Example section)
- The optimization objective ('runtime' or 'quality')
- The termination criterion ('runtime' or 'quality')
- The time limit for target algorithm runs (set to cutoff_time)
- The penalty value for target algorithm runs (if not solving instance within time limit, or crashing)
- How many configurations can be considered as winner in one tournament
- The instance set size
- The size of the subset of instances for the first tournament
- The path to the directory Selector ought to log to
- The time Selector has for the complete AC process

Below you will find a table describing all of Selectors hyperparameters.

.. list-table:: **Hyperparameters for Selector**
   :widths: 30 20 20 80
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - `--file`
     - str
     - ""
     - Path to the file containing selector arguments.
   * - `--check_path`
     - bool
     - False
     - If True, validates the paths passed to Selector.
   * - `--seed`
     - int
     - 42
     - Sets all random generators in Selector.
   * - `--verbosity`
     - int
     - 0
     - Selector prints less, if 0, more if 1.
   * - `--log_folder`
     - str
     - "latest"
     - Path to the directory Selector ought to log to.
   * - `--memory_limit`
     - int
     - 3069
     - Maximum allowed memory for a target algorithm run to use at once.
   * - `--wrapper_mod_name`
     - str
     - ""
     - Name of the target algorithm wrapper. Notation as in module import, e.g. module.sub_module.wrapper.
   * - `--wrapper_class_name`
     - str
     - ""
     - Name of the class in the wrapper module.
   * - `--quality_match`
     - str
     - ""
     - Regex for a line that signifies a solved instance in quality optimization.
   * - `--solve_match`
     - list of str
     - []
     - List of strings that signify a solved instance in runtime optimization, e.g. ['SAT', 'UNSAT'].
   * - `--quality_extract`
     - str
     - ""
     - Regex to extract the objective value.
   * - `--winners_per_tournament`
     - int
     - 1
     - Number of configurations to regard as winners in a tournament.
   * - `--tournament_size`
     - int
     - 5
     - Number of configurations to compete in a tournament.
   * - `--number_tournaments`
     - int
     - 2
     - Number of parallel tournaments.
   * - `--monitor`
     - str
     - "tournament_level"
     - For runtime optimization ("tournament_level" or "instance_level"), level of capping.
   * - `--surrogate_amortized_time`
     - int
     - 30
     - If the GGA model requires more than the set seconds, it will be updated less frequently.
   * - `--termination_criterion`
     - str
     - "runtime"
     - Termination criterion for AC run ("runtime" or "total_tournament_number").
   * - `--total_tournament_number`
     - int
     - 10
     - Total number of tournaments for the main loop.
   * - `--model_update_iteration`
     - int
     - 3
     - Sets the frequency of GGA updates if amortized time violated (# concluded tournaments).
   * - `--generator_multiple`
     - int
     - 5
     - Factor to multiply the number of suggestions from one method.
   * - `--initial_instance_set_size`
     - int
     - 5
     - Instance set size of the first tournaments.
   * - `--set_size`
     - int
     - 50
     - Size of the training instance set.
   * - `--smac_pca_dim`
     - int
     - 8
     - PCA dimension of SMAC.
   * - `--tn`
     - int
     - 100
     - Evaluation history is reduced each time this many tournaments concluded.
   * - `--cleanup`
     - bool
     - False
     - If True, tmp directory is regularly, actively cleaned by selector.
   * - `--cpu_binding`
     - bool
     - False
     - If True, target algorithm and all its child processes are bound to one CPU.
   * - `--scenario_file`
     - str
     - None
     - Path to a file containing hyperparameters concerning the AC scenario, see example.
   * - `--run_obj`
     - str
     - None
     - Optimization metric ("runtime" or "quality").
   * - `--overall_obj`
     - str
     - None
     - PAR to use with the evaluation results.
   * - `--cutoff_time`
     - str
     - None
     - Time limit for a target algorithm to run on one instance.
   * - `--crash_cost`
     - float
     - 10000000
     - Penalty for unfinished instance or crash.
   * - `--wallclock_limit`
     - str
     - None
     - Total amount of time Selector can run.
   * - `--instance_file`
     - str
     - None
     - Path to the file containing the instance paths.
   * - `--feature_file`
     - str
     - None
     - Path to the file containing the instance features.
   * - `--paramfile`
     - str
     - None
     - Path to the parameter space definition of the target algorithm.
   * - `--runtime_feedback`
     - str
     - ""
     - Regex to extract runtime if target algorithm reports it (runtime optimization).


Example Scenario File
---------------------

.. code-block:: python

	paramfile = /your/path/to/params.pcs
	execdir = .
	run_obj = runtime
	overall_obj = PAR10
	cutoff_time = 30
	wallclock_limit = 6400
	instance_file = /your/path/to/instances.txt
	feature_file = /your/path/to/features.txt

Example Selector Args file
--------------------------

.. code-block:: python

	--seed 44
	--par 10

	--winners_per_tournament 1

	--tournament_size 4
	--number_tournaments 2

	--termination_criterion total_runtime
	--monitor tournament_level

	--initial_instance_set_size 5
	--set_size 91

	--generator_multiple 1

	--ta_pid_name ""
	--memory_limit 2048
	--cutoff_time 30

	--check_path False

	--log_folder selector_log
	--wrapper_mod_name selector.your_wrapper
	--wrapper_class_name Your_Wrapper

	You can add whichever arguments you like to these files.