import argparse



class TAP_Work_Wrapper():

    def get_command_line_args(self, runargs, config):

        instance = runargs["instance"]
        id = runargs["id"]
        configuration = f" ".join([f" -{param}={value}" for param, value in config.items() ])

        cmd = f"python -u selector/input/target_algorithms/proxies/tap_work.py {configuration} -i={instance} -ii={id}"

        return cmd

if __name__ == "__main__":
    config = {"c": 2}
    runargs = {"instance": "002"}


    wrapper = TAP_Work_Wrapper()
    print(wrapper.get_command_line_args(runargs, config))