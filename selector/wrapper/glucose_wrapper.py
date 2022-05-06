import os
import argparse

from wrapper.GenericWrapper4AC_master.genericWrapper4AC.domain_specific.satwrapper import SatWrapper
#from wrapper.GenericWrapper4AC_master.domain_specific.satwrapper import SatWrapper


class GLucoseWrapper(SatWrapper):

    def get_command_line_args(self, runargs, config):

        instance = runargs["instance"]
        configuration = f" ".join([f" {param}={value}" for param, value in config.items() ])


        cmd = f'stdbuf -i0 -o0 -e0 ./selector/input/target_algorithms/glucose-master/build/glucose-simp {instance} ' \
              f'-verb=2 -rnd-seed={runargs["seed"]} {configuration}'

        return cmd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=dict)
    parser.add_argument("--runargs", type=dict)

    #config = {'-rnd-freq': '0', '-var-decay': '0.001', '-cla-decay': '0.001', '-gc-frac': '0.000001'}

    #runargs = {'instance': './selector/input/SAT_Instances/modular_kcnf/00541.cnf', 'specifics': 'SAT', 'cutoff': 10,
      #         'runlength': 0, 'seed': 42, 'tmp': '/var/folders/vz/zf_zybt53sg6v1x04sz791s40000gn/T/tmpngro3m22'}

    wrapper = GLucoseWrapper()
    print(wrapper.get_command_line_args(parser.runargs, parser.config))

