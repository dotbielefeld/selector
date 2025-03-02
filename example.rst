.. _example:

Example
=======

You can run selector in Python like this

.. code-block:: python

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
	       wrapper_mod_name='your_path_to.your_wrapper',
	       wrapper_class_name='Your_Wrapper')


You can also call a Python script as exemplified in selector/main.py and pass paths and arguments via command line.

Selector will run the AC process until the 'wallclock_limit' or 'total_tournament_number' is reached and save the results in 'log_folder'.
