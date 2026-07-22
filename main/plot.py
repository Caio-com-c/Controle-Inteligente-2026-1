from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PlotSignals:

    def __init__(self, app):

        self.MAX_POINTS = 120

        self.app = app

        self.window = Toplevel(app.janela)
        self.window.title("Sinais da Simulação")
        self.window.geometry("500x700")

        self.t = []
        self.u = []
        self.y = []
        self.control = []
        self.error = []

        self.fig = Figure(figsize=(6,7), dpi=100)

        self.ax1 = self.fig.add_subplot(411)
        self.ax2 = self.fig.add_subplot(412)
        self.ax3 = self.fig.add_subplot(413)
        self.ax4 = self.fig.add_subplot(414)

        self.ax1.set_title("Entrada / Setpoint")
        self.ax2.set_title("Saída")
        self.ax3.set_title("Sinal de Controle")
        self.ax4.set_title("Erro")

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.grid(True)

        # cria as linhas apenas uma vez
        self.line_u, = self.ax1.plot([], [])
        self.line_y, = self.ax2.plot([], [])
        self.line_c, = self.ax3.plot([], [])
        self.line_e, = self.ax4.plot([], [])

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.window
        )

        #self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.canvas.get_tk_widget().pack(
            fill=BOTH,
            expand=True
        )

        self.fig.tight_layout()


    def update_plot(self):
    
        self.line_u.set_data(self.t, self.u)
        self.line_y.set_data(self.t, self.y)
        self.line_c.set_data(self.t, self.control)
        self.line_e.set_data(self.t, self.error)
    
        if self.t:
        
            xmin = self.t[0]
            xmax = max(self.t[-1], 1)

            if xmin == xmax:
                xmax += 1
    
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(xmin, xmax)
    
        self.canvas.draw_idle()
    

    def add_sample(self, t, u, y, control, error):

        self.t.append(t)
        self.u.append(u)
        self.y.append(y)
        self.control.append(control)
        self.error.append(error)

        if len(self.t) > self.MAX_POINTS:

            self.t.pop(0)
            self.u.pop(0)
            self.y.pop(0)
            self.control.pop(0)
            self.error.pop(0)

        self.update_plot()