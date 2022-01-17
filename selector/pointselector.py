import random


class Point_Selector:

    def __init__(self, pool):
        self.pool = pool

    def select_points(self):
        pass


class Random_Selector(Point_Selector):

    def __init__(self, pool):
        super().__init__(pool)

    def select_points(self, number_of_points):

        selected_ids = random.sample(list(self.pool.configurations), number_of_points)
        self.pool.configurations_for_next_run = {c: self.pool.configurations[c] for c in selected_ids}