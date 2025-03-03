"""In this module the scenario object is constructed."""
import os
import warnings
import argparse
import sys
sys.path.append(os.getcwd())

from selector.read_files import get_ta_arguments_from_pcs, read_instance_paths, read_instance_features


class Scenario:
    """
    Scenario class that stores all relevant information for the configuration

    Parameters
    ----------
    scenario : dict or str
        If string, a scenario file will be read in.
    cmd : dict
        Command line arguments which augment the scenario file/dict.
    """
    def __init__(self, scenario, cmd={'check_path': False}):
            
        if isinstance(scenario, str):
            scenario = self.scenario_from_file(scenario)

        elif isinstance(scenario, dict):
            scenario = scenario

        else:
            raise TypeError("Scenario must be string or dic")

        # add and overwrite cmd line args
        for key, value in cmd.items():

            if key in scenario and value is not None:
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

    def read_scenario_files(self, scenario):
        """
        Read in the relevant files needed for a complete scenario

        Parameters
        ----------
        scenario : dict
            The scenario dictionary.

        Returns
        -------
        scenario : dict
            The updated scenario dictionary.
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

        if self.run_obj not in ["runtime", "quality"]:
            raise ValueError("The specified run objective is not supported")

        if self.overall_obj not in ["mean", "mean10", "PAR10"]:
            raise ValueError("The specified objective is not supported")

        if not isinstance(float(self.cutoff_time), float):
            raise ValueError("The cutoff_time needs to be a float")

        if not isinstance(float(self.wallclock_limit), float):
            raise ValueError("The wallclock_limit needs to be a float")

        # check if the named instances are really available
        if self.check_path:
            for i in (self.instance_set + self.test_instances):
                if not os.path.exists(f"./selector{i}".strip("\n")):
                    raise FileExistsError(f"Instance file {i} does not exist")

        for i in (self.instance_set + self.test_instances):
            if i not in self.features:
                raise ValueError(f"For instance {i} no features were provided")

        if "log_folder" not in list(self.__dict__.keys()):
            setattr(self, "log_folder", "latest")
        elif self.log_folder == "None":
            self.log_folder = "latest"

    def scenario_from_file(self, scenario_path):
        """
        Read in an ACLib scenario file

        Parameters
        ----------
        scenario_path : str
            Path to the scenario file.

        Returns
        -------
        dict
            Dictionary containing the scenario information.
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
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def parse_args():
    """
    Argument parser

    Returns
    -------
    dict
        Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("Hyperparameters of selector")
    so = parser.add_argument_group("Scenario options")

    hp.add_argument('--file', type=open, action=LoadOptionsFromFile)

    #parser.add_argument('--check_path', dest='check_path', action='store_true')
    parser.add_argument('--check_path', default=False, type=lambda x: (str(x).lower() == 'true'))

    hp.add_argument('--seed', default=42, type=int)
    hp.add_argument('--verbosity', default=0, type=int)
    hp.add_argument('--ta_pid_name', type=str, default="")
    hp.add_argument('--log_folder', type=str, default="latest")
    hp.add_argument('--log_location', type=str, default="./selector/logs/")
    hp.add_argument('--memory_limit', type=int, default=1023*3)

    hp.add_argument('--ta_run_type', type=str, default="import_wrapper")
    hp.add_argument('--wrapper_mod_name', type=str, default="")
    hp.add_argument('--wrapper_class_name', type=str, default="")
    hp.add_argument('--quality_match', type=str, default="")
    hp.add_argument('--solve_match', nargs='+', type=str, default=[])
    hp.add_argument('--quality_extract', type=str, default="")

    hp.add_argument('--winners_per_tournament', type=int, default=1)
    hp.add_argument('--tournament_size', type=int, default=5)
    hp.add_argument('--number_tournaments', type=int, default=2)

    hp.add_argument('--par', type=int, default=1)
    hp.add_argument('--monitor', type=str, default="tournament_level")
    hp.add_argument('--surrogate_amortized_time', type=int, default=30)

    hp.add_argument('--termination_criterion', type=str, default="total_runtime")
    hp.add_argument('--total_tournament_number', type=int, default=10)
    hp.add_argument('--model_update_iteration', type=int, default=3)

    hp.add_argument('--generator_multiple', type=int, default=5)
    hp.add_argument('--initial_instance_set_size', type=int, default=5)
    hp.add_argument('--set_size', type=int, default=50)
    hp.add_argument('--instances_dir', type=str, default="")
    hp.add_argument('--smac_pca_dim', type=int, default=8)
    hp.add_argument('--tn', type=int, default=100)
    hp.add_argument('--cleanup', type=bool, default=False)
    hp.add_argument('--cpu_binding', type=bool, default=False)

    so.add_argument('--scenario_file', type=str)
    so.add_argument('--ta_cmd', type=str)
    so.add_argument('--deterministic', type=str)
    so.add_argument('--run_obj', type=str)
    so.add_argument('--overall_obj', type=str)
    so.add_argument('--cutoff_time', type=str)
    so.add_argument('--crash_cost', type=float, default='10000000')
    so.add_argument('--wallclock_limit', type=str)
    so.add_argument('--instance_file', type=str)
    so.add_argument('--feature_file', type=str)
    so.add_argument('--paramfile', type=str)
    so.add_argument('--qual_max', type=bool, default=False)
    so.add_argument('--runtime_feedback', type=str, default='')

    so.add_argument('--w_1', type=float, default=-0.8356679356095191)
    so.add_argument('--w_2', type=float, default=0.8572501015854599)
    so.add_argument('--w_3', type=float, default=0.6037123781208655)
    so.add_argument('--w_4', type=float, default=0.7357117939046054)
    so.add_argument('--w_5', type=float, default=-0.9936592470722203)
    so.add_argument('--w_6', type=float, default=-0.6555611254096128)
    so.add_argument('--w_7', type=float, default=0.6846488604171828)
    so.add_argument('--w_8', type=float, default=-0.29616131125825584)
    so.add_argument('--w_9', type=float, default=-0.42641293642609523)
    so.add_argument('--w_10', type=float, default=-0.636625827266299)
    so.add_argument('--w_11', type=float, default=-0.07705211904734346)
    so.add_argument('--w_12', type=float, default=0.7655061394605909)
    so.add_argument('--w_13', type=float, default=0.07958609777734993)
    so.add_argument('--w_14', type=float, default=-0.6433479035364913)
    so.add_argument('--w_15', type=float, default=0.9330160438280031)
    so.add_argument('--w_16', type=float, default=0.19269241833198825)
    so.add_argument('--w_17', type=float, default=0.924764738193058)
    so.add_argument('--w_18', type=float, default=0.6163122179339903)
    so.add_argument('--w_19', type=float, default=-0.3164211654229764)
    so.add_argument('--w_20', type=float, default=-0.9134468738753907)
    so.add_argument('--w_21', type=float, default=0.09507515338942374)
    so.add_argument('--w_22', type=float, default=-0.12320139765942552)
    so.add_argument('--w_23', type=float, default=0.5334762956128223)
    so.add_argument('--w_24', type=float, default=0.9040962774172033)
    so.add_argument('--w_25', type=float, default=0.7152291148005006)
    so.add_argument('--w_26', type=float, default=0.6264909195211199)
    so.add_argument('--w_27', type=float, default=-0.3078428587569706)
    so.add_argument('--w_28', type=float, default=0.13991257638679933)
    so.add_argument('--w_29', type=float, default=0.6072035854383455)
    so.add_argument('--w_30', type=float, default=0.07142940503717064)
    so.add_argument('--w_31', type=float, default=0.9481229658781036)
    so.add_argument('--w_32', type=float, default=0.260083514586567)
    so.add_argument('--w_33', type=float, default=-0.18358836077393523)
    so.add_argument('--w_34', type=float, default=0.5345236666392377)
    so.add_argument('--w_35', type=float, default=0.7620279025679817)
    so.add_argument('--w_36', type=float, default=-0.19720345812081364)
    so.add_argument('--w_37', type=float, default=0.34488933380271597)
    so.add_argument('--w_38', type=float, default=-0.31776509157739097)
    so.add_argument('--w_39', type=float, default=-0.060524835116740255)
    so.add_argument('--w_40', type=float, default=-0.04932352546335972)
    so.add_argument('--w_41', type=float, default=0.8845189709705542)
    so.add_argument('--w_42', type=float, default=0.07230604126060913)
    so.add_argument('--w_43', type=float, default=-0.6314381312161101)
    so.add_argument('--w_44', type=float, default=-0.2997826811933294)
    so.add_argument('--w_45', type=float, default=0.3920074130279137)
    so.add_argument('--w_46', type=float, default=0.7955255094472115)
    so.add_argument('--w_47', type=float, default=-0.9803992172559881)
    so.add_argument('--w_48', type=float, default=0.8009359266585256)

    return vars(parser.parse_args())


if __name__ == "__main__":

    parser = parse_args()
