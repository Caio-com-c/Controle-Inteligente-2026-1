import control as ctl
import numpy as np

from core.core import Core

class Controle:
    
    def __init__(self):
        # Instância da classe Core
        self.core = Core()

        # Variáveis que serão preenchidas posteriormente
        self.time = None
        self.data_size = 0
        self.plant = None
        self.G = None

    def pre_calc(self, intime, input_signal, output_signal):
        # Armazena o vetor de tempo
        self.time = intime

        # Define a quantidade de amostras
        self.data_size = len(intime)

        # Realiza a estimação da planta
        self.core.estimate(intime, input_signal, output_signal)

        # Obtém os parâmetros estimados da planta
        # Exemplo esperado:
        # plant[0] -> ganho (K)
        # plant[1] -> constante de tempo (tau)
        self.plant = self.core.functionEstimated()

        # G(s) = K / (tau*s + 1)
        # Numerador da planta:
        numerator = [self.plant[0]]

        # Denominador da planta
        denominator = [self.plant[1], 1]

        # Cria a função de transferência da planta
        self.G = ctl.TransferFunction(numerator, denominator)

    def pid(self, set_point=1, kp=0, ki=0, kd=0):
        # Cria o vetor de referência (setpoint constante)
        input_reference = set_point * np.ones(self.data_size)

        # Controlador PID:
        # C(s) = (Kd*s^2 + Kp*s + Ki)/s
        pid_controller = ctl.TransferFunction([kd, kp, ki],[1, 0])

        # Fecha a malha com realimentação unitária
        # T(s) = C(s)G(s) / (1 + C(s)G(s))
        closed_loop = ctl.feedback(pid_controller * self.G, 1)

        # Simula a resposta para a entrada definida
        _, response = ctl.forced_response(closed_loop, T=self.time, U=input_reference)

        # Calcula o erro do PID
        self.pid_erro = response - input_reference
        
        # Retorna a resposta controlada
        return response

    def get_pid_erro(self):
        # Retorna o erro do PID
        return self.pid_erro




