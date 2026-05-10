import control as ctl
import numpy as np

from Core.core import Core
from Data.data import Data

class Controle:
    
    def __init__(self, Data = Data()):
        # Instância da classe Core
        self.core = Core(Data)

        # Variáveis que serão preenchidas posteriormente
        self.time = None
        self.data_size = 0
        self.plant = None
        self.G = None

    def set_pid(self, intime, input_signal, output_signal):
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

    def set_pid_zoh(self, Kp, Ki, Kd, N, Ts, umin=0.0, umax=255.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N
        self.Ts = Ts
        
        # Saturação da saída
        # O umin e umax são os valor limite de saturação do sinal de controle (u)
        # Pode usar os valores min/max da fonte ou min/max de funcionamento da planta
        self.umin = umin
        self.umax = umax

        # Coeficientes do derivativo filtrado
        # ad = e^(-N * Ts)
        self.ad = np.exp(-N * Ts)
        # bd = Kd * N * (1 - ad)
        self.bd = Kd * N * (1.0 - self.ad)

        # Estados internos
        self.integral = 0.0
        self.derivative = 0.0
        self.e_prev = 0.0
        self.pid_zoh_erro = None

    def pid_zoh(self, setpoint, feedback):
        # O setpoint NÃO está em porcentagem

        # Erro
        pid_erro = setpoint - feedback

        # Proporcional
        # P[k] = Kp * pid_erro[k]
        proportional = self.Kp * pid_erro

        # Integral (candidato)
        # Ic[k] = I[k] + Ki * Ts * pid_erro[k]
        integral_candidate = self.integral + self.Ki * self.Ts * pid_erro

        # Erro derivativo
        # derro[k] = pid_erro[k] - pid_erro[k-1]
        derro = pid_erro - self.e_prev

        # Derivativo (candidato)
        # Dc[k] = ad * D[k-1] + bd * derro[k] 
        derivative_candidate = self.ad * self.derivative + self.bd * derro

        # Saída não saturada
        # u_unsat[k] = P[k] + Ic[k] + Dc[k]
        u_unsat = proportional + integral_candidate + derivative_candidate

        # Saturação
        u = max(self.umin, min(self.umax, u_unsat))

        # Anti-windup simples:
        # só aceita a integral se não houver saturação
        if u == u_unsat:
            self.integral = integral_candidate

        # Atualiza estados
        self.derivative = derivative_candidate
        self.e_prev = pid_erro
        self.pid_zoh_erro = pid_erro
        # O sinal de controle (u) varia de acordo com os limites de saturação
        return u

    def get_pid_erro(self):
        # Retorna o erro do PID
        return self.pid_erro
    
    def get_pid_zoh_erro(self):
        # Retorna o erro do PID
        return self.pid_zoh_erro
