class Dados:

    def __init__(self):

        self.clear()

    def clear(self):

        self.t = []
        self.u = []
        self.y = []
        self.control = []
        self.error = []

    def add(self, state):

        self.t.append(state["sim_time"])
        self.u.append(state["sp_pct"])
        self.y.append(state["h_pct"])
        self.control.append(state["u_ctrl"])
        self.error.append(state["error_pct"])

    def get(self):

        return (
            self.t,
            self.control,
            self.y,
            self.u
        )