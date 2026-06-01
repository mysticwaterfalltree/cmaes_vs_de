from pymoo.core.callback import Callback

class History(Callback):

    def __init__(self):
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.opt.get("F")[0])