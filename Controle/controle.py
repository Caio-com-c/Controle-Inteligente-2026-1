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

    def set_pidzoh(self, Kp, Ki, Kd, N, Ts, umin=0.0, umax=255.0):
        """
        Configuração do controlador PID discreto com:
            - Integral por aproximação retangular (ZOH)
            - Derivativo filtrado
    
        Parâmetros:
            Kp   -> ganho proporcional
            Ki   -> ganho integral
            Kd   -> ganho derivativo
            N    -> frequência do filtro derivativo
            Ts   -> período de amostragem (s)
            umin -> limite inferior do sinal de controle
            umax -> limite superior do sinal de controle
        """
    
        # Ganhos do controlador
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
        # Parâmetro do filtro derivativo
        self.N = N
    
        # Tempo entre duas execuções consecutivas do controlador
        self.Ts = Ts
    
        # Limites físicos do atuador
        # Exemplo:
        # PWM Arduino: 0 a 255
        # Tensão: 0 a 12 V
        self.umin = umin
        self.umax = umax
        
        self.ad = np.exp(-N * Ts)
    
        # Ganho aplicado à diferença do erro
        self.bd = Kd * N * (1.0 - self.ad)
    
        # ==================================================
        # Estados internos do controlador
        # ==================================================
    
        # Estado acumulado da ação integral
        self.integral = 0.0
    
        # Estado interno do filtro derivativo
        self.derivative = 0.0
    
        # Erro da amostra anterior
        self.e_prev = 0.0
    
        # Armazena o último erro calculado
        self.pid_zoh_erro = None

    def pidzoh(self, setpoint, feedback):
        """
        Executa uma iteração do PID discreto.
    
        Entradas:
            setpoint -> valor desejado
            feedback -> valor medido
    
        Saída:
            u -> sinal de controle
        """
    
        # ==================================================
        # Cálculo do erro
        # ==================================================
        #
        # erro = referência - medição
        #
        pid_erro = setpoint - feedback
    
        # ==================================================
        # Parcela Proporcional
        # ==================================================
        #
        # P[k] = Kp * erro[k]
        #
        proportional = self.Kp * pid_erro
    
        # ==================================================
        # Parcela Integral
        # ==================================================
        #
        # Aproximação discreta:
        #
        # I[k] = I[k-1] + Ki*Ts*erro[k]
        #
        # Ainda não atualizamos o estado real.
        # Primeiro calculamos um candidato.
        #
        integral_candidate = (
            self.integral +
            self.Ki * self.Ts * pid_erro
        )
    
        # ==================================================
        # Diferença do erro
        # ==================================================
        #
        # de[k] = erro[k] - erro[k-1]
        #
        derro = pid_erro - self.e_prev
    
        # ==================================================
        # Derivativo filtrado
        # ==================================================
        #
        # D[k] = ad*D[k-1] + bd*de[k]
        #
        derivative_candidate = (
            self.ad * self.derivative +
            self.bd * derro
        )
    
        # ==================================================
        # Soma das ações de controle
        # ==================================================
        #
        # u = P + I + D
        #
        u_unsat = (
            proportional +
            integral_candidate +
            derivative_candidate
        )
    
        # ==================================================
        # Saturação do atuador
        # ==================================================
        #
        # Garante que o sinal enviado para a planta
        # permaneça dentro dos limites permitidos.
        #
        u = max(self.umin, min(self.umax, u_unsat))
    
        # ==================================================
        # Anti-Windup
        # ==================================================
        #
        # Se houve saturação, não atualizamos a integral.
        #
        # Isso evita que a integral continue crescendo
        # enquanto o atuador está no limite.
        #
        if u == u_unsat:
            self.integral = integral_candidate
    
        # ==================================================
        # Atualização dos estados
        # ==================================================
        self.derivative = derivative_candidate
        self.e_prev = pid_erro
        self.pid_zoh_erro = pid_erro
    
        return u
    
    def set_pid_autotune(self, Ts, d=13334, N=20.0, u=13334):
        """
        Inicializa o autotune por relé
        (Método de Åström-Hägglund).
    
        Ts -> período de amostragem
        d  -> amplitude do relé
        N  -> filtro derivativo futuro
        u  -> valor máximo esperado da variável medida
        """
    
        self.Ts = Ts
        self.d = d
        self.N = N
    
        # Indica que o controlador ainda está
        # na fase de sintonia.
        self.tuning = True
    
        # Relógio interno
        self.t = 0.0
    
        # Máximo e mínimo observados
        self.y_max = -u
        self.y_min = u
    
        # Armazena instantes dos cruzamentos
        self.cross_times = []
    
        # Sinal inicial do erro
        self.sign_prev = 1

    def pid_autotune(self, setpoint, feedback):
        if self.tuning:
            return self._autotune(setpoint, feedback)
        else:
            return self._pid(setpoint, feedback)

    def _autotune(self, setpoint, y):
        e = setpoint - y
        self.t += self.Ts

        # Atualiza máximos e mínimos
        if y > self.y_max:
            self.y_max = y
        if y < self.y_min:
            self.y_min = y

        # Sinal do erro
        sign = 1 if e >= 0 else -1

        # Detecta cruzamento por zero
        if sign != self.sign_prev:
            self.cross_times.append(self.t)
            self.sign_prev = sign

        # Após 6 cruzamentos (3 ciclos), calcula os ganhos
        if len(self.cross_times) >= 6:
            self._calculate_gains()

        # Saída do relé
        return self.d if sign >= 0 else -self.d

    # Calcula ganhos do PID
    def _calculate_gains(self):
        # Amplitude da oscilação
        a = (self.y_max - self.y_min) / 2.0

        # Período médio
        periods = []
        for i in range(2, len(self.cross_times)):
            periods.append(self.cross_times[i] - self.cross_times[i - 2])

        Pu = np.mean(periods)

        # Ganho crítico
        Ku = 4.0 * self.d / (np.pi * a)

        # Ziegler-Nichols
        self.Kp = 0.6 * Ku
        self.Ki = 2.0 * self.Kp / Pu
        self.Kd = self.Kp * Pu / 8.0

        # Coeficientes do derivativo filtrado
        self.ad = np.exp(-self.N * self.Ts)
        self.bd = self.Kd * self.N * (1.0 - self.ad)

        # Zera estados
        self.I = 0.0
        self.D = 0.0
        self.e_prev = 0.0

        # Entra em modo PID
        self.tuning = False
    
    def _pid(self, setpoint, y):
        e = setpoint - y

        # Proporcional
        P = self.Kp * e

        # Integral
        self.I += self.Ki * self.Ts * e

        # Derivativo filtrado
        de = e - self.e_prev
        self.D = self.ad * self.D + self.bd * de

        # Atualiza erro anterior
        self.e_prev = e

        # Saída do controlador
        return P + self.I + self.D
    
    def get_pid_erro(self):
        # Retorna o erro do PID
        return self.pid_erro
    
    def get_pid_zoh_erro(self):
        # Retorna o erro do PID
        return self.pid_zoh_erro
