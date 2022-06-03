import argparse



class TAP_Sleep_Wrapper():

    def get_command_line_args(self, runargs, config):

        instance = runargs["instance"]
        configuration = f" ".join([f" -{param}={value}" for param, value in config.items() ])

        cmd = f"python -u selector/input/target_algorithms/proxies/tap_sleep.py {configuration} -i={instance}"

        return cmd

if __name__ == "__main__":
    config = {"c": 2}
    runargs = {"instance": "002"}


    wrapper = TAP_Sleep_Wrapper()
    print(wrapper.get_command_line_args(runargs, config))