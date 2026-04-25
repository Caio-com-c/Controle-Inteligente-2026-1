import numpy as np
import control as ctr
import random

class Core:

    def __init__(self, Data):
        # Inicializa as dependências
        self.data = Data
        
        # Inicializa Tau como 0
        self.tau = None

        # Obtém os dados e converte para NumPy
        self.time = self.data.t().to_numpy()          # Vetor de tempo
        self.output = self.data.H().to_numpy()        # Saída medida
        self.input_signal = self.data.Qin().to_numpy()# Entrada

        # Valores de regime permanente (último ponto)
        self.final_output = self.output[-1]           # Saída final
        self.final_input = self.input_signal[-1]      # Entrada final

        print(self.tau)


    def estimate(self):
        # Calcula 63.2% do valor final
        target = 0.632 * self.final_output

        # Procura o ponto onde a saída atinge 63.2%
        for i, y in enumerate(self.output):
            if y >= target:
                self.tau = self.time[i-1]  # Tempo correspondente
                break

        # Gera resposta estimada usando modelo exponencial
        estimated_output = self.final_output * (1 - np.exp(-self.time / self.tau))

        return estimated_output

    def functionEstimated(self):
        # Verifica se a constante de tempo foi definida
        if self.tau is None:
            print(
                "Aviso: o valor de tau não foi definido. "
                "O sistema não foi estimado."
            )
            return None

        # Calcula o ganho estático do sistema
        gain = self.final_output / self.final_input

        # Cria a função de transferência de primeira ordem:
        transfer_function = np.array([gain, self.tau])

        return transfer_function

    def addNoise(self, undisturbed, disturb=0):
        # Gera ruído gaussiano
        noise = [random.gauss(0, disturb) for _ in undisturbed]
        # Adiciona ruído à saída original
        noisy_output = [y + n for y, n in zip(undisturbed, noise)]
        
        return noisy_output
