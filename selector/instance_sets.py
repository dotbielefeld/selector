"""In this module the choice of instance sets for tournaments is handled."""
import numpy as np
import math

__all__ = ['TimedInstanceSet']


class InstanceSet:
    """
    Selects the next instance set to run a tournament on.

    Parameters
    ----------
    instance_set : list
        Set of instances available.
    start_instance_size : int
        Size of the first instances se to be created.
    set_size : int
        If not set, the biggest instance set to be created includes all instances in training instance_set. If set, the biggest instance set will be of the size of the given int.
    instance_increment_size : int
        Number of instances to add to the next instance set.
    target_reach : int
        If set, Selector will try to reach full instance set size after this many tournaments.
    """
    def __init__(self, instance_set, start_instance_size, set_size=None, instance_increment_size=None, target_reach=None):
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.instance_sets = []
        self.subset_counter = 0

        if set_size and set_size <= len(instance_set):
            self.set_size = set_size
        else:
            self.set_size = len(instance_set)

        if instance_increment_size:
            self.instance_increment_size = instance_increment_size
        else:
            self.instance_increment_size = self.set_size / np.floor(self.set_size/self.start_instance_size)

        if target_reach:
            set_size = self.start_instance_size
            counter = 0
            while set_size <= self.set_size:
                set_size += self.instance_increment_size
                counter += 1
            self.target_increment = np.ceil(target_reach / counter)
        else:
            self.target_increment = 1

    def next_set(self):
        """
        Create a new instance set with instances not included in any created
        set before. This is not thread safe and should only be called from the
        master node.
        """
        # Get instances that were already chosen
        if self.subset_counter == 0:
            seen_instances = []
        else:
            seen_instances = self.instance_sets[self.subset_counter - 1]

        # If we have not chosen any instance before we create a first set
        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                new_subset = np.random.choice(self.instance_set, self.start_instance_size, replace=False).tolist()
                self.instance_sets.append(new_subset)
        # If we have seen instances before and are still allowed to choose instances we do so
        elif len(seen_instances) <= self.set_size:
            # We either sample as many instances as the slope tells us or the last remainder set to full instance sice
            number_to_sample = int(min(self.instance_increment_size, self.set_size - len(seen_instances)))
            # We can only select instances not chosen before
            possible_instances = [i for i in self.instance_set if i not in seen_instances]
            new_subset = np.random.choice(possible_instances, number_to_sample, replace=False).tolist()
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1] + new_subset)
        # In case we have chosen all instances but still tournaments to run we use the full set of instances for all
        # following tournaments
        else:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])

        self.subset_counter += 1

    def get_subset(self, next_tournament_set_id):
        """
        Create an instance set for the next tournament. The set contains all
        instances that were included in the previous sets as well as a new
        subset of instances.

        Parameters
        ----------
        next_tournament_set_id : int
            Id of the subset to get the next instances for .
        Returns
        -------
        tuple
            - int, id of the instances set.
            - list, next instance set.
        """
        assert next_tournament_set_id <= self.subset_counter
        # If we have already created the required subset we return it
        if next_tournament_set_id in range(len(self.instance_sets)):
            next_set = self.instance_sets[next_tournament_set_id]

        elif self.target_increment != 1 and next_tournament_set_id % self.target_increment != 0:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])
            next_set = self.instance_sets[next_tournament_set_id - 1]
            self.subset_counter += 1

        # In case we have not, we create the next instance subset
        else:
            self.next_set()
            next_set = self.instance_sets[next_tournament_set_id]
        return next_tournament_set_id, next_set


class FlexInstanceSet(InstanceSet):
    """
    Selects the next instance set to run a tournament on.

    Parameters
    ----------
    instance_set : list
        Set of instances available.
    start_instance_size : int
        Size of the first instances se to be created.
    set_size : int
        If not set, the biggest instance set to be created includes all instances in training instance_set. If set, the biggest instance set will be of the size of the given int.
    instance_increment_size : int
        Number of instances to add to the next instance set.
    target_reach : int
        If set, Selector will try to reach full instance set size after this many tournaments.
    target_start : int
        If set, Selector will only start increasing instance size after this number tournament.
    """
    def __init__(self, instance_set, start_instance_size, set_size=None, instance_increment_size=None, target_reach=None, target_start=None):
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.instance_sets = []
        self.subset_counter = 0
        self.target_start = target_start

        if set_size and set_size <= len(instance_set):
            self.set_size = set_size
        else:
            self.set_size = len(instance_set)

        if instance_increment_size:
            self.instance_increment_size = instance_increment_size
        else:
            self.instance_increment_size = self.set_size / np.floor(self.set_size / self.start_instance_size)

        if target_reach:
            set_size = self.start_instance_size
            counter = 0
            while set_size <= self.set_size:
                set_size += self.instance_increment_size
                counter += 1
            if target_start:
                self.target_increment = np.ceil(((target_reach - target_start) / self.set_size))
            else:
                self.target_increment = np.ceil(target_reach / counter)
        else:
            self.target_increment = 1

    def next_set(self):
        """
        Create a new instance set with instances not included in any created
        set before. This is not thread safe and should only be called from the
        master node.
        """
        # Get instances that were already chosen
        if self.subset_counter == 0:
            seen_instances = []
        else:
            seen_instances = self.instance_sets[self.subset_counter - 1]

        # If we have not chosen any instance before we create a first set
        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                new_subset = np.random.choice(self.instance_set, self.start_instance_size, replace=False).tolist()
                self.instance_sets.append(new_subset)
        # If we have seen instances before and are still allowed to choose instances we do so
        elif len(seen_instances) <= self.set_size:
            # We either sample as many instances as the slope tells us or the last remainder set to full instance sice
            number_to_sample = int(min(self.instance_increment_size, self.set_size - len(seen_instances)))
            # We can only select instances not chosen before
            possible_instances = [i for i in self.instance_set if i not in seen_instances]
            if self.target_start:
                if self.subset_counter >= self.target_start:
                    new_subset = np.random.choice(possible_instances, number_to_sample, replace=False).tolist()
                else:
                    new_subset = []
            else:
                new_subset = np.random.choice(possible_instances, number_to_sample, replace=False).tolist()
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1] + new_subset)
        # In case we have chosen all instances but still tournaments to run we use the full set of instances for all
        # following tournaments
        else:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])

        self.subset_counter += 1


class TimedInstanceSet(InstanceSet):
    """
    Selects the next instance set to run a tournament on.

    Parameters
    ----------
    instance_set : list
        Set of instances available.
    start_instance_size : int
        Size of the first instances se to be created.
    set_size : int
        If not set, the biggest instance set to be created includes all instances in training instance_set. If set, the biggest instance set will be of the size of the given int.
    runtime : int
        Maximum runtime of the AC process.
    start_time : float
        Fracture of the Maximum runtime when increasing the set size starts.
    end_time : int
        Fracture of the Maximum runtime when maximum set size ought to be reached. Depending on the scenario, this cannot be quaranteed.
    """
    def __init__(self, instance_set, start_instance_size, set_size=None,
                 runtime=42, start_time=0.1, end_time=0.7):
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.instance_sets = []
        self.subset_counter = 0
        self.runtime = runtime
        self.start_time = start_time
        self.end_time = end_time
        self.instance_increment_size = 0
        self.target_increment = 1

        if set_size and set_size <= len(instance_set):
            self.set_size = set_size
        else:
            self.set_size = len(instance_set)

    def next_set(self):
        """
        Create a new instance set with instances not included in any created
        set before. This is not thread safe and should only be called from the
        master node.
        """
        # Get instances that were already chosen
        if self.subset_counter == 0:
            seen_instances = []
        else:
            seen_instances = self.instance_sets[self.subset_counter - 1]

        # If we have not chosen any instance before we create a first set
        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                new_subset = np.random.choice(self.instance_set, self.start_instance_size, replace=False).tolist()
                self.instance_sets.append(new_subset)
        # If we have seen instances before and are still allowed to choose instances we do so
        elif len(seen_instances) <= self.set_size:
            # We either sample as many instances as the slope tells us or the last remainder set to full instance sice
            number_to_sample = int(min(self.instance_increment_size, self.set_size - len(seen_instances)))
            # We can only select instances not chosen before
            possible_instances = [i for i in self.instance_set if i not in seen_instances]
            if possible_instances:
                new_subset = np.random.choice(possible_instances, number_to_sample, replace=False).tolist()
                self.instance_sets.append(self.instance_sets[self.subset_counter - 1] + new_subset)
            else:
                self.instance_sets.append(self.instance_sets[self.subset_counter - 1])
        # In case we have chosen all instances but still tournaments to run we use the full set of instances for all
        # following tournaments
        else:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])

        self.subset_counter += 1

    def get_subset(self, next_tournament_set_id, time, iteration):
        """
        Create an instance set for the next tournament. The set contains all
        instances that were included in the previous sets as well as a new
        subset of instances.

        Parameters
        ----------
        next_tournament_set_id : int
            Id of the subset to get the next instances for.
        time : int
            Currently passed time since start of the AC process.
        iteration : int
            Last iteration that was finished.
        Returns
        -------
        tuple
            - int, id of the instance set.
            - list, next instance set.
        """
        assert next_tournament_set_id <= self.subset_counter

        if time >= self.start_time * self.runtime:
            caution = ((1 - (self.set_size - len(self.instance_sets[self.subset_counter - 1])) / self.set_size)) + 1
            time_per_iteration = (time / iteration) * caution
            max_it = max((((self.runtime * self.end_time) - time) / time_per_iteration) * (1 - (time / self.runtime)) * self.end_time, 1)
            if max_it * self.instance_increment_size / self.target_increment < self.set_size - len(self.instance_sets[self.subset_counter - 1]):
                while max_it * self.instance_increment_size / self.target_increment < self.set_size - len(self.instance_sets[self.subset_counter - 1]) and self.set_size != len(self.instance_sets[self.subset_counter - 1]):    
                    if self.target_increment > 1:
                        self.target_increment -= 1
                    elif self.instance_increment_size < int(self.set_size * ((self.set_size - len(self.instance_sets[self.subset_counter - 1])) / max_it)):
                        self.instance_increment_size += 1
                        # Making sure we don't increase more than 2% of complete instance set
                        if self.instance_increment_size > int(self.set_size * 0.02):
                            break
                    else:
                        break
            elif max_it * self.instance_increment_size / self.target_increment >= self.set_size - len(self.instance_sets[self.subset_counter - 1]):
                while max_it * self.instance_increment_size / self.target_increment >= self.set_size - len(self.instance_sets[self.subset_counter - 1]) and self.set_size != len(self.instance_sets[self.subset_counter - 1]):
                    if self.instance_increment_size > 1:
                        self.instance_increment_size -= 1
                    else:
                        self.target_increment += 1

        # If we have already created the required subset we return it
        if next_tournament_set_id in range(len(self.instance_sets)):
            next_set = self.instance_sets[next_tournament_set_id]

        elif self.target_increment != 1 and next_tournament_set_id % self.target_increment != 0:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])
            next_set = self.instance_sets[next_tournament_set_id - 1]
            self.subset_counter += 1

        # In case we have not, we create the next instance subset
        else:
            self.next_set()
            next_set = self.instance_sets[next_tournament_set_id]

        return next_tournament_set_id, next_set
