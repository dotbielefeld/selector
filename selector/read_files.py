import re
from selector.pool import Parameter

import numpy as np

def get_ta_arguments_from_pcs(para_file):
    """
     Read in a file that contains the target algorithm parameters. The file is .pcs and adheres to the structure of
     AClib.
    :param para_file: str. Path to the .pcs file
    :return: list, list. A list containing information on the parameters of the target algorithm & a list containing
    parameter value assignments that are not possible
    """

    no_goods = []
    parameters = []
    conditionals = {}

    with open(para_file, 'r') as pf:
        for line in pf:
            line = line.strip().split("#", 1)[0]

            # skip empty lines
            if line == "":
                continue

            line_split = line.split(" ", 1) # TODO should be whitespace, as the pcs could have a tab, etc.
            param_name = line_split[0]
            param_info = line_split[1] # TODO This will error if forbidden params do not have any spaces
            param_info = re.sub('\s+', ' ', param_info) # TODO not sure why we need to do this; check

            if "|" not in param_info:

                # cat
                if re.search(r'\{', param_info): # TODO change to not use re module
                    type , bounds, defaults = get_categorical(param_name, param_info)
                    parameters.append(Parameter(param_name, type, bounds, defaults, {}, ''))
                # forbidden
                elif  re.search(r'\{', param_name):
                    no_good = get_no_goods(line)
                    no_goods.append(no_good)
                # cont.
                elif re.search(r'\[', param_info):
                    type, bounds, defaults, scale = get_continuous(param_name, param_info)
                    parameters.append(Parameter(param_name, type, bounds, defaults, {}, scale))

            # conditionals
            elif "|" in param_info:
                condition_param, condition = get_conditional(param_name, param_info)

                if param_name not in conditionals:
                    conditionals[param_name] = {condition_param: condition}
                else:
                    conditionals[param_name].update({condition_param: condition})

            else:
                raise ValueError("The parameter file contains unreadable elements. Check that the structure adheres"
                                 "to AClib") # TODO Specify more clearly where we have an issue

    # adding conditionals to parameters

    for pc in conditionals:
        condition_found = False
        for parameter in parameters:
            if pc in parameter.name:
                parameter.condition.update(conditionals[pc])
                condition_found = True

        if not condition_found:
            raise ValueError("A condition was parsed that does not correspond to a read parameter of the"
                             " target algorithm")

    return parameters,  no_goods, conditionals


def get_categorical(param_name, param_info):
    """
    For a categorical parameter: check if its parsed attributes are valid and extract information on the parameter
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: type , bounds, defaults of the parameter
    """

    bounds = re.search(r'.*\{(.*)\}', param_info).group().strip("{ }").split(",") # Remove first .* in re?

    defaults = re.findall(r'\[(.*)\]*]', param_info)

    if bounds[0] in ["yes", "no", "on", "off"]: # TODO need to ensure bounds[1] is also reasonable, also len(bounds) == 2
        type = "boolean" # TODO should just be cat

        if defaults[0] in ["on", "yes"]:
            defaults = True
        elif defaults[0] in [ "no", "off"]:
            defaults = False
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not within [yes, no, on, off]")

        bounds = [b in ["on", "yes"] for b in bounds]

    elif isinstance(float(bounds[0]), float):
        type = "cat."

        if isinstance(float(defaults[0]), float):
            defaults = float(defaults[0])
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not categorical")

        bounds = [float(b) for b in bounds]

    else:
        raise ValueError(f"For parameter {param_name} the parsed bounds were not boolean or categorical")

    return type , bounds, defaults

def get_continuous(param_name, param_info):
    """
    For a continuous parameter: check if its parsed attributes are valid and extract information on the parameter
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: type , bounds, defaults,scale of the parameter
    """

    scale = re.search(r'[a-zA-Z]+', param_info)
    param_info = re.findall(r'\[[^\]]*]', param_info)
    bounds = param_info[0].strip("[] ").split(",")
    defaults = param_info[1].strip("[] ")

    # checking for set scale
    if scale and "i" in scale.group():
        type = "int"
        scale = scale.group().strip("i")

        if isinstance(int(defaults), int):
            defaults = int(defaults)
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not integer")

        bounds = [int(b) for b in bounds]

    else:
        type = "cont."

        if isinstance(float(defaults), float):
            defaults = float(defaults)
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not continuous")
        bounds = [float(b) for b in bounds]

        if scale is None:
            scale = ''
        else:
            scale = scale.group()

    return type, bounds, defaults, scale

def get_conditional(param_name, param_info):
    """
    For a parameter: get the information on conditionals
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: condition_param, condition
    """
    param_info = param_info.strip(" | ")

    condition = re.search(r'\{(.*)\}', param_info).group().strip("{ }").split(",")
    condition_param = re.search(r'.+?(?=in)', param_info).group()

    if condition[0] in ["yes", "no", "on", "off"]:
        condition = [c in ["on", "yes"] for c in condition]
    elif isinstance(float(condition[0]), float):
        condition = [float(c) for c in condition]    
    else:
        raise ValueError(f"For parameter {param_name} the parsed conditions could not be read")

    return condition_param, condition

def get_no_goods(no_good):
    """
    Takes an string of form: {param_1=value_1 , param_2=value_2, ...} and returns a dic of the no good
    :param no_good: str. Takes an string of form: {param_1=value_1 , param_2=value_2, ...}
    :return: dic
    """

    forbidden = {}
    no_good = no_good.strip("{ }").split(",")

    for ng in no_good:
        param, value = ng.split("=")
        param = param.strip()
        value = value.strip()

        if value in ["yes", "on"]:
            value = True
        elif value in ["no", "off"]:
            value = False
        elif isinstance(float(value), float):
            value = float(value)
        else:
            raise ValueError(f"For no good {no_good} the parameter values are not known")

        forbidden[param] = value

    return forbidden


def read_instance_paths(instance_set_path):
    """
    Read in instances from an AClib instance file
    :param instance_set_path: str. Path to the instance file
    :return: list. List of paths to the instances
    """

    instance_set = []

    with open(instance_set_path, 'r') as f:
        for line in f:
            instance_set.append(line.strip())

    return instance_set

def read_instance_features(feature_set_path):
    """
    Read in features from an AClib features file
    :param feature_set_path: str. Path to the feature file
    :return: dic, list. Dic with the read in features list with the feature names
    """

    features = {}
    with open(feature_set_path, 'r') as f:
        lines = f.readlines()
        feature_names = lines[0].strip().split(",")[1:]

        for line in lines[1:]:
            line = line.strip().split(",")
            features[line[0]] = np.array(line[1:], dtype=np.single)

    return features, feature_names
