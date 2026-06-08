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

    def set_system_objects(self, controle_obj, planta_obj, setpoint=1.0):
        
        #Injeta os objetos configurados externamente e define o setpoint desejado.
        
        self.controle_obj = controle_obj
        self.planta_obj = planta_obj
        self.setpoint = setpoint

    def update_signals(self):
        #Executa a iteração dinâmica em malha fechada entre o controlador e a planta.
        if self.controle_obj is None or self.planta_obj is None:
            return

        # 1. Calcula o erro atual da malha (e = r - y)
        self.erro_signal = self.setpoint - self.planta_signal

        # 2. Injeta o erro no controlador escolhido pelo usuário para obter o sinal 'u'
        if hasattr(self.controle_obj, 'tuning'):
            #print(f"[PLOT] Modo Ativo: AUTOTUNE | Executando: pid_autotune | Erro: {self.erro_signal:.4f}")
            self.controle_signal = self.controle_obj.pid_autotune(self.setpoint, self.planta_signal)
        else:
            #print(f"[PLOT] Modo Ativo: PID ZOH   | Executando: pidzoh       | Erro: {self.erro_signal:.4f}")
            self.controle_signal = self.controle_obj.pidzoh(self.setpoint, self.planta_signal)

        # 3. Alimenta a Planta de Primeira Ordem com o sinal de controle 'u'
        self.planta_signal = self.planta_obj.order_one(self.controle_signal)

    def update_plot(self):

        self.t += self.dt

        # Calcula a iteração atual da malha dinâmica antes de atualizar a tela
        self.update_signals()

        # Captura os sinais gerados
        controle = self.controle_signal
        planta = self.planta_signal
        erro = self.erro_signal

        # Adiciona novos pontos
        self.x_data.append(self.t)

        self.y_controle.append(controle)
        self.y_planta.append(planta)
        self.y_erro.append(erro)

        # Mantém apenas a janela desejada
        while (
            len(self.x_data) > 0 and
            self.x_data[0] < self.t - self.janela_tempo
        ):

            self.x_data.pop(0)

            self.y_controle.pop(0)
            self.y_planta.pop(0)
            self.y_erro.pop(0)

        # Atualiza linhas
        self.line_controle.set_data(
            self.x_data,
            self.y_controle
        )

        self.line_planta.set_data(
            self.x_data,
            self.y_planta
        )

        self.line_erro.set_data(
            self.x_data,
            self.y_erro
        )

        # Atualiza eixo do tempo
        if self.t < self.janela_tempo:

            self.ax.set_xlim(
                0,
                self.janela_tempo
            )

        else:

            self.ax.set_xlim(
                self.t - self.janela_tempo,
                self.t
            )

        self.canvas.draw_idle()
