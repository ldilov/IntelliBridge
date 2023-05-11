from abc import ABC, abstractmethod


class AbstractHub(ABC):
    @abstractmethod
    def download(self, repo_or_name, dir_name):
        pass