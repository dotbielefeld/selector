import os
import warnings
import argparse
import pickle
import sys
sys.path.append(os.getcwd())

from selector.read_files import get_ta_arguments_from_pcs, read_instance_paths, read_instance_features





class Scenario:

    def __init__(self, scenario, cmd={'check_path': False}):
        """
        Scenario class that stores all relevant information for the configuration
        :param scenario: dic or string. If string, a scenario file will be read in.
        :param cmd: dic, Command line arguments which augment the scenario file/dic
        """
        
        if isinstance(scenario, str):
            scenario = self.scenario_from_file(scenario)

        elif isinstance(scenario, dict):
            scenario = scenario

        else:
            raise TypeError("Scenario must be string or dic")

        # add and overwrite cmd line args
        for key, value in cmd.items():

            if key in scenario and value != None:
                warnings.warn(f"Setting: {key} of the scenario file is overwritten by parsed command line arguments")
                scenario[key] = value

            elif key not in scenario:
                scenario[key] = value

        self.read_scenario_files(scenario)

        for arg_name, arg_value in scenario.items():
            setattr(self, arg_name, arg_value)

        self.cutoff_time = float(self.cutoff_time)
        self.wallclock_limit = float(self.wallclock_limit)

        self.verify_scenario()

        with open(f'./selector/logs/{self.log_folder}/scenario.pkl', 'wb') as out:
            pickle.dump(scenario, out)



    def read_scenario_files(self, scenario):

        """
        Read in the relevant files needed for a complete scenario
        :param scenario: dic.
        :return: scenario: dic.
        """

        # read in
        if "paramfile" in scenario:
            scenario["parameter"], scenario["no_goods"], scenario["conditionals"] = get_ta_arguments_from_pcs(scenario["paramfile"])
        else:
            raise ValueError("Please provide a file with the target algorithm parameters")

        if "instance_file" in scenario:
            scenario["instance_set"] = read_instance_paths(scenario["instance_file"])
        else:
            raise ValueError("Please provide a file with the training instances")

        if "test_instance_file" in scenario:
            scenario["test_instances"] = read_instance_paths(scenario["test_instance_file"])
        else:
            scenario["test_instances"] = []

        if "feature_file" in scenario:
            scenario["features"], scenario["feature_names"] = read_instance_features(scenario["feature_file"])
        else:
            raise ValueError("Please provide a file with instance features")

        return scenario

    def verify_scenario(self):
        """
        Verify that the scenario attributes are valid
        """
        # TODO: verify algo and execdir

        if self.run_obj not in ["runtime"]:
            raise ValueError("The specified run objective is not supported")

        if self.overall_obj not in ["mean", "mean10", "PAR10"]:
            raise ValueError("The specified objective is not supported")

        if not isinstance(float(self.cutoff_time), float) :
            raise ValueError("The cutoff_time needs to be a float")

        if not isinstance(float(self.wallclock_limit), float) :
            raise ValueError("The wallclock_limit needs to be a float")

        # check if the named instances are really available
        if self.check_path:
            for i in (self.instance_set + self.test_instances):
                if not os.path.exists(f".{i}".strip("\n")):
                    raise FileExistsError(f"Instance file {i} does not exist")

        for i in (self.instance_set + self.test_instances):
            if i not in self.features:
                raise ValueError(f"For instance {i} no features were provided")

        if  "log_folder" not in list(self.__dict__.keys()):
            setattr(self, "log_folder" , "latest")
        elif self.log_folder == "None":
            self.log_folder = "latest"



    def scenario_from_file(self, scenario_path):

        """
        Read in an ACLib scenario file
        :param scenario_path: Path to the scenario file
        :return: dic containing the scenario information
        """

        name_map = {"algo": "ta_cmd"}
        scenario_dict = {}

        with open(scenario_path, 'r') as sc:
            for line in sc:
                line = line.strip()

                if "=" in line:

                    #remove comments
                    pairs = line.split("#", 1)[0].split("=")
                    pairs = [l.strip(" ") for l in pairs]

                    # change of AClib names to names we use. Extend name_map if necessary
                    if pairs[0] in name_map:
                        key = name_map[pairs[0]]
                    else:
                        key = pairs[0]

                    scenario_dict[key] = pairs[1]
        return scenario_dict

class LoadOptionsFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def parse_args():
    """
    Argparser
    :return: dic. Dic of arguments parsed
    """

    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("Hyperparameters of selector")
    so = parser.add_argument_group("Scenario options")

    hp.add_argument('--file', type=open, action=LoadOptionsFromFile)

    hp.add_argument('--check_path', default=False)
    hp.add_argument('--seed', default=42)
    hp.add_argument('--ta_pid_name', type=str, default="")
    hp.add_argument('--log_folder', type=str, default="latest")
    hp.add_argument('--memory_limit', type=int, default=1023*3)

    hp.add_argument('--ta_run_type', type=str, default="import_wrapper")
    hp.add_argument('--wrapper_mod_name', type=str, default="", required='--ta_run_type' in sys.argv)
    hp.add_argument('--wrapper_class_name', type=str, default="", required='--ta_run_type' in sys.argv)

    hp.add_argument('--winners_per_tournament', type=int, default=1)
    hp.add_argument('--tournament_size', type=int, default=5 )
    hp.add_argument('--number_tournaments', type=int, default=2)

    hp.add_argument('--par', type=int, default=1)

    hp.add_argument('--termination_criterion', type=str, default="runtime")
    hp.add_argument('--total_tournament_number', type=int, default=10)

    hp.add_argument('--generator_multiple', type=int, default=5)
    hp.add_argument('--initial_instance_set_size', type=int, default=5)
    hp.add_argument('--set_size', type=int, default=50)

    so.add_argument('--scenario_file', type=str)
    so.add_argument('--ta_cmd', type=str)
    so.add_argument('--deterministic', type=str)
    so.add_argument('--run_obj', type=str)
    so.add_argument('--overall_obj', type=str)
    so.add_argument('--cutoff_time', type=str)
    so.add_argument('--wallclock_limit', type=str)
    so.add_argument('--instance_file', type=str)
    so.add_argument('--feature_file', type=str)
    so.add_argument('--paramfile', type=str)

    return vars(parser.parse_args())

if __name__ == "__main__":

    parser = parse_args()
