import numpy as np
import sys
import math
from selector.pool import Configuration


# Check for conditionals and turn of parameters if violated
def satisfy_conditionals(s,config_setting):

    conf_to_del = []

    # Child node is turned off if parent node does not take value in specified range
    for child_node in s.conditionals:

        for parent_node in s.conditionals[child_node]:

            for params in s.parameter:

                if params.name.replace(" ", "") == parent_node.replace(" ", ""):

                    parent_info = s.conditionals[child_node]

                    param_space = params

                    continue

            for parent in parent_info:

                if param_space.type == 'cat.' or param_space.type == 'boolean':    

                    if config_setting[parent_node.replace(" ", "")] in parent_info[parent_node]:
                        pass

                    else:
                        if child_node not in conf_to_del:

                            conf_to_del.append(child_node)

                elif param_space.type == 'int' or param_space.type == 'cont.':

                    if parent_info[parent_node][0] \
                    <= config_setting[parent_node.replace(" ", "")] <= parent_info[parent_node][1]:
                        pass

                    else:
                        if child_node not in conf_to_del:

                            conf_to_del.append(child_node)

        for ctd in conf_to_del:

            config_setting.pop(ctd, None)

    return config_setting

# Check for no goods and reset parameter values if violated
def satisfy_no_goods(s,config_setting):

    for ng in s.no_goods:

    	violation = True

    	while violation:

            ng_values = list(ng.values())

            config_set_values = []

            for ng_element in ng:

                config_set_values.append(config_setting[ng_element])

            if config_set_values == ng_values:

                configs_to_reset = []

                violation = True

                for params in s.parameter:

                    if params.name.replace(" ", "") in ng:

                        configs_to_reset.append(params)

                new_setting = random_set_conf(configs_to_reset)

                for ns in new_setting:

                    config_setting[ns] = new_setting[ns]

            else:
                violation = False

    return config_setting



# Generating random configuration values for given param space
def random_set_conf(parameter):
    
    config_setting = {}

    for param in parameter:

        config_setting[param.name] = 1

        if param.type == 'int':

            if param.scale == 'l':

                # Generate in logarithmic space
                config_setting[param.name] \
                = int(math.exp(np.random.randint(low=math.log(param.bound[0]),
                	                             high=math.log(param.bound[1]))))
            else:
                config_setting[param.name] = np.random.randint(low=param.bound[0],
                	                                           high=param.bound[1])

        elif param.type == 'cont.':

            if param.scale == 'l':

                # Generate in logarithmic space
                config_setting[param.name] \
                = math.exp(np.random.uniform(low=math.log(param.bound[0]),
                	                         high=(param.bound[1])))
            else:        	
                config_setting[param.name] = np.random.uniform(low=param.bound[0],
                	                                           high=param.bound[1])

        elif param.type == 'cat.':

            config_setting[param.name] = np.random.choice(param.bound)

        elif param.type == 'boolean':

            config_setting[param.name] = np.random.choice(param.bound)

    return config_setting



def generate(s):

    # Unique configuration ID
    ID = np.random.randint(low=(-sys.maxsize - 1),high=sys.maxsize)

    # initialize gender
    gender = np.random.choice(['C','N'])

    # Generate configuration randomly based on given parameter space
    config_setting = random_set_conf(s.parameter)

    # Check conditionals and turn of parameters if violated
    config_setting = satisfy_conditionals(s,config_setting)

    # Check no goods and reset values if violated
    config_setting = satisfy_no_goods(s,config_setting)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(ID, config_setting, gender)

    return configuration