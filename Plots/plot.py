import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal as scipy_signal
from Core.core import Core
from Controle.controle import Controle

plt.ion()


class Plot(Controle):

    def __init__(self, parent_frame=None):
        Controle.__init__(self)
        self.parent_frame = parent_frame
        self.noise_level = 0.05
        self.filter_cutoff = 0.5
        self.filter_order = 2
        self.time = self.G = self.plant = self.core = None

        self._fig = None
        self._ax  = None

        plt.style.use('dark_background')
        self.colors = {
            'signal':     '#4A90D9',
            'reference':  '#5DADE2',
            'text':       '#85C1E9',
            'grid':       '#1a2a3a',
            'grid_major': '#1a3a5a',
            'grid_minor': '#0a1a2a',
        }

    # ── Setters ──────────────────────────────────────────────────────────────

    def set_core_data(self, data_instance, G=None):
        self.core = Core(data_instance)
        self.time = data_instance.t()

        if G is not None:
            self.G = G

        try:
            # Garante que tau foi calculado
            self.core.estimate()

            g = self.core.functionEstimated()  # [gain, tau]

            if g is not None:
                ganho, tau = g
                self.plant = np.array([ganho, tau])

                if self.G is None:
                    import control as ct
                    s = ct.tf('s')
                    self.G = ganho / (tau*s + 1)

        except Exception:
            pass

    def set_system_data(self, time, G, plant):
        self.time, self.G, self.plant = time, G, plant

    def set_parent_frame(self, frame):
        self.parent_frame = frame

    def set_noise_level(self, level):
        try:
            self.noise_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self.noise_level = 0.05

    def set_filter(self, cutoff=None, order=None):
        if cutoff is not None:
            self.filter_cutoff = max(0.01, min(1.0, float(cutoff)))
        if order is not None:
            self.filter_order = max(1, int(order))

    # ── Signal processing ────────────────────────────────────────────────────

    def _add_noise(self, sig, noise_on, noise_level=None):
        if not noise_on:
            return sig
        level = noise_level if noise_level is not None else self.noise_level
        return sig + np.random.normal(0, level, len(sig))

    def _apply_filter(self, data, filter_on, cutoff=None, order=None):
        if not filter_on or len(data) == 0 or self.time is None or len(self.time) < 2:
            return data
        fs = 1.0 / (self.time[1] - self.time[0])
        normalized = min(0.99, (cutoff or self.filter_cutoff) / (fs / 2))
        if not (0 < normalized < 1):
            return data
        b, a = scipy_signal.butter(order or self.filter_order, normalized, btype='low')
        return scipy_signal.filtfilt(b, a, data)

    # ── Figura persistente ───────────────────────────────────────────────────

    def _get_fig_ax(self):
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._fig = plt.figure("Simulação PID", figsize=(10, 5), facecolor='#0a0a1a')
            self._ax  = self._fig.add_subplot(111)
        return self._fig, self._ax

    # ── Estilo ───────────────────────────────────────────────────────────────

    def _setup_axes(self, ax):
        ax.set_facecolor('#0a1428')
        ax.tick_params(colors=self.colors['text'], which='both')
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
        ax.grid(True, color=self.colors['grid_major'], linestyle='-', alpha=0.4, linewidth=0.8)
        ax.grid(True, which='minor', color=self.colors['grid_minor'], linestyle='-', alpha=0.3, linewidth=0.5)
        ax.minorticks_on()

    def _style_legend(self, ax):
        legend = ax.legend(loc='best', facecolor='#0a1428',
                           edgecolor=self.colors['text'], fontsize=10)
        for text in legend.get_texts():
            text.set_color(self.colors['text'])

    def _label_axes(self, ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, color=self.colors['text'], fontsize=11)
        ax.set_ylabel(ylabel, color=self.colors['text'], fontsize=11)
        ax.set_title(title,   color=self.colors['text'], fontsize=13, fontweight='bold')

    def _text_box(self, ax, x, y, text, color, ha='left'):
        ax.text(x, y, text, transform=ax.transAxes, color=color, fontsize=9,
                verticalalignment='top', horizontalalignment=ha,
                bbox=dict(boxstyle='round', facecolor='#0a1428', alpha=0.8))

    # ── Plot PID ─────────────────────────────────────────────────────────────

    def plot_response(self, set_point=1, kp=0, ki=0, kd=0,
                      noise_on=False, filter_on=False,
                      noise_level=None, filter_cutoff=None,
                      title=None, linewidth=2):

        if self.time is None or self.G is None:
            print("Erro: execute 'Calcular' antes de plotar.")
            return None

        try:
            response = self.pid(set_point, kp, ki, kd)
        except Exception as e:
            print(f"Erro ao executar PID: {e}")
            return None

        if response is None:
            return None

        response = self._add_noise(response, noise_on, noise_level)
        response = self._apply_filter(response, filter_on, filter_cutoff)

        response_pct = response * 100
        setpoint_pct = set_point * 100

        fig, ax = self._get_fig_ax()
        ax.cla()
        fig.set_facecolor('#0a0a1a')
        self._setup_axes(ax)

        ax.plot(self.time, response_pct, self.colors['signal'],
                linewidth=linewidth, label='Resposta do Sistema')
        ax.fill_between(self.time, 0, response_pct, alpha=0.1, color=self.colors['signal'])

        ax.plot(self.time, np.full_like(self.time, setpoint_pct),
                self.colors['reference'], linewidth=linewidth,
                linestyle='--', label=f'Setpoint ({setpoint_pct:.0f}%)', alpha=0.9)

        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))

        if title is None:
            title = f"PID: Kp={kp}, Ki={ki}, Kd={kd}"
            if noise_on:
                title += f" | Ruído: σ={noise_level or self.noise_level}"
            if filter_on:
                title += f" | Filtro: fc={filter_cutoff or self.filter_cutoff} Hz"
            if not noise_on and not filter_on:
                title += " | Sem Ruído/Filtro"

        self._label_axes(ax, 'Tempo (s)', 'Altura (%)', title)
        self._style_legend(ax)

        if self.plant is not None:
            self._text_box(ax, 0.02, 0.98,
                           f"G(s) = {self.plant[0]:.3f} / ({self.plant[1]:.3f}s + 1)",
                           self.colors['text'])

        self._text_box(ax, 0.98, 0.98,
                       f"Kp={kp}  Ki={ki}  Kd={kd}",
                       self.colors['reference'], ha='right')

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)

        return response_pct

    # ── Plot identificação ───────────────────────────────────────────────────

    def plot_identification(self, title="Identificação do Sistema", linewidth=2):

        if self.core is None or self.core.time is None or self.core.output is None:
            print("Erro: Core não inicializado ou sem dados.")
            return

        fig, ax = self._get_fig_ax()
        ax.cla()
        fig.set_facecolor('#0a0a1a')
        self._setup_axes(ax)

        ax.plot(self.core.time, self.core.output, self.colors['signal'],
                linewidth=linewidth, label='Saída Real')

        ax.plot(self.core.time, self.core.estimate(), self.colors['reference'],
                linewidth=linewidth, linestyle='--', label='Saída Estimada')

        self._label_axes(ax, 'Tempo (s)', 'Saída', title)
        self._style_legend(ax)

        if self.plant is not None:
            self._text_box(ax, 0.02, 0.98,
                           f"G(s) = {self.plant[0]:.3f} / ({self.plant[1]:.3f}s + 1)",
                           self.colors['text'])

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)
