from matplotlib.figure import Figure

class Plot:
    def __init__(self, figure=None):
        self.fig = figure if figure is not None else Figure()
        self.axes = []
        self.lines = []
        self.current_ax = None

    def subplot(self, n_rows, n_cols, index):
        ax = self.fig.add_subplot(n_rows, n_cols, index)
        self.axes.append(ax)
        self.current_ax = ax

    def create_line(self, color="k", label=None):
        line, = self.current_ax.plot([], [], color, label=label)
        self.lines.append(line)
        return line

    def update_line(self, line, x, y):
        line.set_data(x, y)
        self.current_ax.relim()
        self.current_ax.autoscale_view()

    def _apply_labels(self, x_label, y_label, title):
        self.current_ax.set_xlabel(x_label)
        self.current_ax.set_ylabel(y_label)
        self.current_ax.set_title(title)

    def grid(self):
        self.current_ax.grid()

    def clear(self):
        self.fig.clf()
        self.axes = []
        self.lines = []