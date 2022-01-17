from dataclasses import dataclass, field

from typing import Dict


@dataclass
class Configuration:
    id: int
    conf: list
    meta_data: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    intermidate_results: dict = field(default_factory=dict)

    def add(self, place: str, id, value):
        assert place not in ["meta_data", "history", "results", "intermidate_results"], "Not a valid method"
        getattr(self, place)[id] = value


@dataclass
class Pool:
    configurations: Dict[int, Configuration] = field(default_factory=dict)
    configurations_for_next_run: Dict[int, Configuration] = field(default_factory=dict)

    def add_configuration(self, conf):
        self.configurations[conf.id] = conf

    def get_confguration(self, id):
        return self.configurations[id]