import numpy as np

class Metrics:
    def __init__(self, t, u, y, r=None):
        """
        t : vetor de tempo
        u : sinal de controle
        y : saída do sistema
        r : referência (default = degrau no valor final de y)
        """
        self.t = np.array(t)
        self.u = np.array(u)
        self.y = np.array(y)

        if r is None:
            self.r = np.ones_like(y) * y[-1]
        else:
            self.r = np.array(r)

        self.e = self.r - self.y  # erro

    # ---------------------------
    # ERRO EM REGIME PERMANENTE
    # ---------------------------
    def erro_regime(self):
        return self.e[-1]

    # ---------------------------
    # IAE - Integral do Erro Absoluto
    # ---------------------------
    def IAE(self):
        return np.trapz(np.abs(self.e), self.t)

    # ---------------------------
    # ISE - Integral do Erro Quadrático
    # ---------------------------
    def ISE(self):
        return np.trapz(self.e**2, self.t)

    # ---------------------------
    # Energia do sinal de controle
    # ---------------------------
    def energia_controle(self):
        return np.trapz(self.u**2, self.t)

    # ---------------------------
    # Sobressinal (%OS)
    # ---------------------------
    def sobressinal(self):
        y_final = self.y[-1]
        y_max = np.max(self.y)

        if y_final == 0:
            return 0

        return ((y_max - y_final) / abs(y_final)) * 100

    # ---------------------------
    # Tempo de acomodação (Ts)
    # Critério: 2%
    # ---------------------------
    def tempo_acomodacao(self, tol=0.02):
        y_final = self.y[-1]
        limite_sup = y_final * (1 + tol)
        limite_inf = y_final * (1 - tol)

        for i in range(len(self.y)):
            if np.all((self.y[i:] >= limite_inf) & (self.y[i:] <= limite_sup)):
                return self.t[i]

        return np.nan  # não acomodou

    # ---------------------------
    # Tempo de subida (Tr)
    # 10% -> 90%
    # ---------------------------
    def tempo_subida(self):
        y_final = self.y[-1]

        y_10 = 0.1 * y_final
        y_90 = 0.9 * y_final

        t10 = None
        t90 = None

        for i in range(len(self.y)):
            if t10 is None and self.y[i] >= y_10:
                t10 = self.t[i]
            if t90 is None and self.y[i] >= y_90:
                t90 = self.t[i]

        if t10 is not None and t90 is not None:
            return t90 - t10

        return np.nan

    # ---------------------------
    # RESUMO GERAL
    # ---------------------------
    def resumo(self):
        return {
            "Erro regime permanente": self.erro_regime(),
            "IAE": self.IAE(),
            "ISE": self.ISE(),
            "Energia controle": self.energia_controle(),
            "Sobressinal (%)": self.sobressinal(),
            "Tempo de acomodação (Ts)": self.tempo_acomodacao(),
            "Tempo de subida (Tr)": self.tempo_subida()
        }
