import numpy as np


class InstanceSet:
    def __init__(self, instance_set, start_instance_size, max_set_size=None):
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.instance_sets = {}
        if max_set_size:
            self.max_set_size = max_set_size
        else:
            self.max_set_size = len(instance_set)


    def compute_next_set(self, id):
        """
        Create a new instance set with instances not included in any created set before
        :param id: int. Id of the new instance set. If it is the first set id will be 1.
        """

        # Get instances that were already chosen
        seen_instances = [i for v in self.instance_sets.values() for i in v]

        # If we have not chosen any instance before we create a first set
        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                self.instance_sets[1]  = np.random.choice(self.instance_set, self.start_instance_size, replace=False)
        # If we have seen instances before are still allowed to choose instances we do so
        elif len(seen_instances) < self.max_set_size:
            # We either sample as many instances as the slope tells us or the last remainder set to full instance sice
            number_to_sample = int(min(np.floor(len(self.instance_set)/self.start_instance_size),
                                       self.max_set_size - len(seen_instances)))
            # We can only select instances not chosen before
            possible_instances = [i for i in self.instance_set if i not in seen_instances]
            self.instance_sets[id] = np.random.choice(possible_instances, number_to_sample, replace=False)


    def get_instance_sub_set(self, previous_set_id):
        """
        Create an instance set for the next tournament. The set contains all instances that were included in the
        previous sets as well as a new subset of instances.
        :param previous_set_id: int. Id of the subset of instances used before.
        :return: id of the instances set, list containing instances and previous instances of the subset
        """
        next_set_id = previous_set_id + 1
        # If we have already created the required subset we return it
        if next_set_id in self.instance_sets:
            next_set = [i for id in range(1,next_set_id+1) for i in self.instance_sets[id]]
        # In case we have not, we create the next instance subset
        else:
            self.compute_next_set(next_set_id)
            # We have to make sure that when creating the instance set only the subsets with an id less or equal
            # to the next id are included.
            if next_set_id+1 in self.instance_sets.keys():
                next_set = [i for id in range(1,next_set_id+1) for i in self.instance_sets[id]]
            else:
                next_set = [i for id in range(1, len(self.instance_sets.keys()) + 1) for i in self.instance_sets[id]]
        return next_set_id, next_set





