"""In this module a function for translating configurations is defined."""


def config_translator(s, config):
    """
    Converts boolean values back to their original representation.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    config : selector.pool.Configuration
        Configuration class object.

    Returns
    -------
    dict
        Parameter configuration with converted boolean values.
    """

    boolean1 = ["on", "off"]
    boolean2 = ["yes", "no"]

    for param in s.parameter:
        if not param.original_bound == []:

            if param.original_bound == boolean1:
                if config.conf[param.name]:
                    config.conf[param.name] = "on"
                else:
                    config.conf[param.name] = "off"

            if param.original_bound == boolean2:
                if config.conf[param.name]:
                    config.conf[param.name] = "yes"
                else:
                    config.conf[param.name] = "no"

    return config
