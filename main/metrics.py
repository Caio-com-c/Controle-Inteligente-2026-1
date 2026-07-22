import numpy as np


class Metrics:
    def __init__(self, t, u, y, r=None):
        """
        t : vetor de tempo
        u : sinal de controle
        y : saída do sistema
        r : referência (setpoint)
        """

        self.t = np.asarray(t, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.y = np.asarray(y, dtype=float)

        if len(t) < 2:
            raise ValueError("Dados insuficientes para cálculo das métricas.")

        if r is None:
            self.r = np.ones_like(self.y) * self.y[-1]
        else:
            self.r = np.asarray(r, dtype=float)

        self.e = self.r - self.y

        self.calcular_metricas()

    # ======================================================
    # Cálculo geral
    # ======================================================

    def calcular_metricas(self):

        self.erro_regime = self._erro_regime()
        self.iae = self._iae()
        self.ise = self._ise()
        self.itae = self._itae()
        self.itse = self._itse()

        self.energia_controle = self._energia_controle()
        self.pico_controle = self._pico_controle()

        self.sobressinal = self._sobressinal()

        self.tempo_subida = self._tempo_subida()
        self.tempo_acomodacao = self._tempo_acomodacao()

        self.erro_rms = self._erro_rms()

        self.valor_final = self.y[-1]
        self.valor_maximo = np.max(self.y)

    # ======================================================
    # Erros
    # ======================================================

    def _erro_regime(self):
        return self.e[-1]

    def _erro_rms(self):
        return np.sqrt(np.mean(self.e**2))

    # ======================================================
    # Índices integrais
    # ======================================================

    def _iae(self):
        return np.trapz(np.abs(self.e), self.t)

    def _ise(self):
        return np.trapz(self.e**2, self.t)

    def _itae(self):
        return np.trapz(self.t * np.abs(self.e), self.t)

    def _itse(self):
        return np.trapz(self.t * self.e**2, self.t)

    # ======================================================
    # Controle
    # ======================================================

    def _energia_controle(self):
        return np.trapz(self.u**2, self.t)

    def _pico_controle(self):
        return np.max(np.abs(self.u))

    # ======================================================
    # Resposta temporal
    # ======================================================

    def _sobressinal(self):

        y_final = self.y[-1]

        if abs(y_final) < 1e-12:
            return 0.0

        return max(0.0, (np.max(self.y) - y_final) / abs(y_final) * 100)

    def _tempo_subida(self):

        y_final = self.y[-1]

        y10 = 0.1 * y_final
        y90 = 0.9 * y_final

        t10 = None
        t90 = None

        for i, valor in enumerate(self.y):

            if t10 is None and valor >= y10:
                t10 = self.t[i]

            if t90 is None and valor >= y90:
                t90 = self.t[i]
                break

        if t10 is None or t90 is None:
            return np.nan

        return t90 - t10

    def _tempo_acomodacao(self, tol=0.02):

        y_final = self.y[-1]

        limite_inf = y_final * (1 - tol)
        limite_sup = y_final * (1 + tol)

        for i in range(len(self.y)):

            if np.all(
                (self.y[i:] >= limite_inf) &
                (self.y[i:] <= limite_sup)
            ):
                return self.t[i]

        return np.nan

    # ======================================================
    # Resumo
    # ======================================================

    def resumo(self):

        return {
            "Erro de regime permanente": self.erro_regime,
            "Erro RMS": self.erro_rms,
            "IAE": self.iae,
            "ISE": self.ise,
            "ITAE": self.itae,
            "ITSE": self.itse,
            "Energia de controle": self.energia_controle,
            "Pico do controle": self.pico_controle,
            "Valor final": self.valor_final,
            "Valor máximo": self.valor_maximo,
            "Sobressinal (%)": self.sobressinal,
            "Tempo de subida (Tr)": self.tempo_subida,
            "Tempo de acomodação (Ts)": self.tempo_acomodacao,
        }