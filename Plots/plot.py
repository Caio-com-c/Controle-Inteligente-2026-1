import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton
)

from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from controle import Controle
from plant import Plant
from core import Core


class Plot(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Osciloscópio")
        self.resize(1000, 600)

        # CONFIGURAÇÕES

        self.janela_tempo = 10  # segundos
        self.setpoint = 1.0     # Valor desejado padrão

        # Objetos do sistema (serão injetados externamente)
        self.controle_obj = None
        self.planta_obj = None
        self.core = Core()

        # FIGURA

        self.fig = Figure(facecolor="#e9ecef")
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.ax = self.fig.add_subplot(111)

        self.ax.set_facecolor("#f8f9fa")

        self.ax.grid(
            True,
            color="#d0d7de",
            linewidth=0.8
        )

        for spine in self.ax.spines.values():
            spine.set_color("#adb5bd")

        self.ax.tick_params(colors="#495057")

        self.ax.set_title(
            "Resposta do Sistema",
            color="#212529"
        )

        self.ax.set_xlabel(
            "Tempo (s)",
            color="#343a40"
        )

        self.ax.set_ylabel(
            "Amplitude",
            color="#343a40"
        )

        self.ax.set_xlim(0, self.janela_tempo)
        self.ax.set_ylim(-1.5, 2.5)

        # DADOS

        self.x_data = []

        self.y_controle = []
        self.y_planta = []
        self.y_erro = []

        # LINHAS

        self.line_controle, = self.ax.plot(
            [],
            [],
            color="#0d6efd",
            linewidth=2,
            label="Sinal de Controle"
        )

        self.line_planta, = self.ax.plot(
            [],
            [],
            color="#dc3545",
            linewidth=2,
            label="Planta Controlada"
        )

        self.line_erro, = self.ax.plot(
            [],
            [],
            color="#198754",
            linewidth=2,
            label="Erro"
        )

        self.ax.legend(frameon=False)

        # CANAIS
        
        self.channels = [
            {
                "line": self.line_controle,
                "buffer": self.y_controle,
                "getter": lambda: self.controle_signal,
            },
            {
                "line": self.line_planta,
                "buffer": self.y_planta,
                "getter": lambda: self.planta_signal,
            },
            {
                "line": self.line_erro,
                "buffer": self.y_erro,
                "getter": lambda: self.erro_signal,
            },
        ]

        # BOTÕES

        self.bt_ch1 = QPushButton("CH1 - Sinal de Controle")
        self.bt_ch2 = QPushButton("CH2 - Planta Controlada")
        self.bt_ch3 = QPushButton("CH3 - Erro")

        for bt in [self.bt_ch1, self.bt_ch2, self.bt_ch3]:
            bt.setCheckable(True)
            bt.setChecked(True)

        self.bt_ch1.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
        """)

        self.bt_ch2.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
        """)

        self.bt_ch3.setStyleSheet("""
            QPushButton {
                background: #198754;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
        """)

        self.bt_ch1.clicked.connect(
            lambda: self.toggle_channel(self.line_controle)
        )

        self.bt_ch2.clicked.connect(
            lambda: self.toggle_channel(self.line_planta)
        )

        self.bt_ch3.clicked.connect(
            lambda: self.toggle_channel(self.line_erro)
        )

        # LAYOUT

        buttons_layout = QHBoxLayout()

        buttons_layout.addWidget(self.bt_ch1)
        buttons_layout.addWidget(self.bt_ch2)
        buttons_layout.addWidget(self.bt_ch3)

        main_layout = QVBoxLayout()

        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

        # TIMER

        self.t = 0.0
        self.dt = 0.05
        self.controle_signal = 0.0
        self.planta_signal = 0.0
        self.erro_signal = 0.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(self.dt * 1000))

    def toggle_channel(self, line):

        line.set_visible(
            not line.get_visible()
        )

        self.canvas.draw_idle()

    def set_system_objects(self, controle_obj, planta_obj, setpoint=1.0, noise=0):

        # Injeta os objetos configurados externamente e define setpoint
        self.controle_obj = controle_obj
        self.planta_obj = planta_obj

        self.set_setpoint(setpoint)
        self.set_noise(noise)

    def set_setpoint(self, novo_setpoint, ajustar_escala=True):
        
        # Atualiza o setpoint em tempo real, SEM reiniciar a simulação
        # Pode ser chamado a qualquer momento, de fora da classe
        
        self.setpoint = novo_setpoint

        if not ajustar_escala:
            return

        if novo_setpoint > 0:
            # Se o setpoint for positivo, dá uma folga para baixo e para cima
            self.ax.set_ylim(-novo_setpoint * 1, novo_setpoint * 2.5)
        elif novo_setpoint < 0:
            # Se o setpoint for negativo, ajusta a lógica dos limites
            self.ax.set_ylim(novo_setpoint * 1.5, -novo_setpoint * 0.2)
        else:
            # Se for zero, mantém o padrão antigo
            self.ax.set_ylim(-1.5, 2.5)

    def set_controller(self, novo_controle_obj):
        # Troca o controlador ativo em tempo real, SEM reiniciar a simulação
        self.controle_obj = novo_controle_obj

    def set_noise(self, novo_noise):
        # Troca o nível de ruído aplicado à planta em tempo real, sem reiniciar a simulação
        self.noise = novo_noise

    def update_signals(self):
        # Executa a iteração dinâmica em malha fechada entre o controlador e a planta
        if self.controle_obj is None or self.planta_obj is None:
            return

        # 1. Calcula o erro atual da malha (e = r - y)
        self.erro_signal = self.setpoint - self.planta_signal

        # 2. Injeta as leituras no método correspondente do controlador ativo
        if hasattr(self.controle_obj, 'simulador'):
            # Se o controlador foi configurado em modo Fuzzy, executa-o
            self.controle_signal = self.controle_obj.ControladorFuzzy(self.setpoint, self.planta_signal)

        elif hasattr(self.controle_obj, 'tuning'):
            # Se possuir a tag 'tuning', roda o Autotune por Relé
            self.controle_signal = self.controle_obj.pid_autotune(self.setpoint, self.planta_signal)

        else:
            # Caso contrário, assume o PID ZOH tradicional
            self.controle_signal = self.controle_obj.pidzoh(self.setpoint, self.planta_signal)

        # 3. Alimenta a Planta de Primeira Ordem com o sinal de controle 'u'
        self.planta_signal = self.planta_obj.order_one(self.controle_signal)
        self.planta_signal = self.core.addNoise(self.planta_signal, self.noise)

    def update_plot(self):
        # Chamada a cada "tick" do QTimer
        self.t += self.dt

        # Calcula a iteração atual da malha dinâmica antes de atualizar a tela
        self.update_signals()

        # Adiciona o novo ponto de tempo
        self.x_data.append(self.t)

        # Alimenta o buffer de CADA canal com o valor atual do sinal
        for ch in self.channels:
            ch["buffer"].append(ch["getter"]())

        # Corte da janela de tempo
        limite = self.t - self.janela_tempo
        corte = next(
            (i for i, x in enumerate(self.x_data) if x >= limite),
            len(self.x_data)
        )
        if corte > 0:
            self.x_data = self.x_data[corte:]
            for ch in self.channels:
                del ch["buffer"][:corte]

        # Atualiza cada linha com seus dados já processados
        for ch in self.channels:
            ch["line"].set_data(self.x_data, ch["buffer"])

        # Atualiza eixo do tempo
        self.ax.set_xlim(
            max(0, self.t - self.janela_tempo),
            max(self.t, self.janela_tempo)
        )

        self.canvas.draw_idle()
