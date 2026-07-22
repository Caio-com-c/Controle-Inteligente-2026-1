"""
fuzzy_controller.py — Controlador Fuzzy Mamdani aprimorado
===========================================================

Melhorias em relação à versão anterior
---------------------------------------
1. 7 labels em vez de 5 → maior resolução nas transições:
   NB, NM, NS, ZE, PS, PM, PB
   (Grande / Médio / Pequeno — Negativo e Positivo)

2. Funções de pertinência gaussianas → transições mais suaves e
   sem "saltos" que as triangulares causavam em baixo erro.

3. Terceira entrada: integral do erro (Σe) normalizada → o
   controlador passa a ter "memória" de acumulação de offset,
   eliminando erro em regime permanente sem precisar de um PID-I.

4. Tabela de regras 7×7 = 49 regras (apenas sobre erro + Δe);
   a integral atua como fator de ponderação extra na saída.

5. Defuzzificação por centroide com grade de 500 pontos
   (vs 200 anterior) → mais precisa com custo computacional ainda
   adequado para dt = 0.1 s.

Arquitetura
-----------
Entradas  : erro e [%], Δe [%], Σe (integral normalizada)
Saída     : ΔQe — incremento da vazão [cm³/s]
            Qe[k] = clip(Qe[k-1] + ΔQe, 0, Qe_max)
"""

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
#  Funções de pertinência
# ────────────────────────────────────────────────────────────────────────────

def _gaussiana(x: float, centro: float, sigma: float) -> float:
    """Função de pertinência gaussiana normalizada em [0,1]."""
    return float(np.exp(-0.5 * ((x - centro) / sigma) ** 2))


def _gaussiana_vec(x: np.ndarray, centro: float, sigma: float) -> np.ndarray:
    """Versão vetorizada para pré-computar sobre a grade."""
    return np.exp(-0.5 * ((x - centro) / sigma) ** 2)


# ────────────────────────────────────────────────────────────────────────────
#  Classe principal
# ────────────────────────────────────────────────────────────────────────────

class FuzzyController:
    """
    Controlador Fuzzy Mamdani com 7 labels gaussianas e 3 entradas.

    Parâmetros
    ----------
    Qe_max        : float — Vazão máxima da bomba [cm³/s].
    sigma_entrada : float — Largura das gaussianas de entrada (padrão 0.18).
    sigma_saida   : float — Largura das gaussianas de saída  (padrão 0.18).
    ganho_integral: float — Peso da integral do erro na saída (padrão 0.15).
    n_pontos      : int   — Resolução da grade de defuzzificação (padrão 500).
    """

    # 7 labels: NB NM NS ZE PS PM PB
    # Centros normalizados em [-1, 1]
    _CENTROS = np.array([-1.0, -0.667, -0.333, 0.0, 0.333, 0.667, 1.0])
    _N       = 7

    # Índices de label
    NB, NM, NS, ZE, PS, PM, PB = 0, 1, 2, 3, 4, 5, 6

    # Tabela de regras 7×7 [erro_idx][delta_e_idx] → saida_idx
    # Lógica: quanto maior o erro e maior a velocidade de crescimento → maior saída
    _REGRAS = [
        # Δe:  NB   NM   NS   ZE   PS   PM   PB
        [NB,   NB,  NB,  NM,  NM,  NS,  ZE],   # e = NB
        [NB,   NM,  NM,  NM,  NS,  ZE,  PS],   # e = NM
        [NB,   NM,  NS,  NS,  ZE,  PS,  PM],   # e = NS
        [NM,   NM,  NS,  ZE,  PS,  PM,  PM],   # e = ZE
        [NM,   NS,  ZE,  PS,  PS,  PM,  PB],   # e = PS
        [NS,   ZE,  PS,  PM,  PM,  PM,  PB],   # e = PM
        [ZE,   PS,  PM,  PM,  PB,  PB,  PB],   # e = PB
    ]

    def __init__(
        self,
        Qe_max: float,
        sigma_entrada: float = 0.18,
        sigma_saida: float   = 0.18,
        ganho_integral: float = 0.15,
        n_pontos: int = 500,
    ) -> None:
        self.Qe_max         = float(Qe_max)
        self.sigma_entrada  = float(sigma_entrada)
        self.sigma_saida    = float(sigma_saida)
        self.ganho_integral = float(ganho_integral)
        self.n_pontos       = int(n_pontos)

        # Grade de saída normalizada [-1, 1]
        self._x_saida = np.linspace(-1.0, 1.0, n_pontos)

        # Pré-computa MFs de saída sobre a grade (7 arrays)
        self._mf_saida = [
            _gaussiana_vec(self._x_saida, c, sigma_saida)
            for c in self._CENTROS
        ]

        # Estado interno: Σe acumulada (integrador fuzzy)
        self._integral_e = 0.0

    # ------------------------------------------------------------------ #
    #  Interface pública                                                  #
    # ------------------------------------------------------------------ #

    def compute(self, erro_pct: float, delta_erro_pct: float, dt: float = 0.1) -> float:
        """
        Calcula o incremento ΔQe.

        Parâmetros
        ----------
        erro_pct       : erro (SP − h) em % de H_max  [−100, +100]
        delta_erro_pct : Δe = e[k] − e[k−1]           [−100, +100]
        dt             : passo de tempo [s] para acumular integral

        Retorna
        -------
        float : ΔQe em [−Qe_max, +Qe_max]
        """
        # Atualiza integral (acumulada, limitada para não explodir)
        self._integral_e += erro_pct * dt
        self._integral_e  = float(np.clip(self._integral_e, -500.0, 500.0))

        # Normaliza entradas para [-1, 1]
        e_n  = float(np.clip(erro_pct       / 100.0, -1.0, 1.0))
        de_n = float(np.clip(delta_erro_pct / 100.0, -1.0, 1.0))
        # Integral normalizada (div por valor típico de saturação)
        ie_n = float(np.clip(self._integral_e / 500.0, -1.0, 1.0))

        # Fuzzificação
        mu_e  = self._fuzzificar(e_n)
        mu_de = self._fuzzificar(de_n)

        # Inferência Mamdani (49 regras)
        agregado = np.zeros(self.n_pontos)
        for i, row in enumerate(self._REGRAS):
            for j, saida_idx in enumerate(row):
                forca = min(mu_e[i], mu_de[j])
                if forca > 1e-6:
                    agregado = np.maximum(
                        agregado,
                        np.minimum(forca, self._mf_saida[saida_idx])
                    )

        # Defuzzificação — centroide
        soma = np.sum(agregado)
        if soma < 1e-9:
            centroide = 0.0
        else:
            centroide = float(np.sum(self._x_saida * agregado) / soma)

        # Componente proporcional (regras) + componente integral
        saida_norm = centroide + self.ganho_integral * ie_n
        saida_norm = float(np.clip(saida_norm, -1.0, 1.0))

        return saida_norm * self.Qe_max

    def reset(self) -> None:
        """Reinicia a integral interna (ao mudar de modo ou setpoint)."""
        self._integral_e = 0.0

    # ------------------------------------------------------------------ #
    #  Auxiliares privados                                                 #
    # ------------------------------------------------------------------ #

    def _fuzzificar(self, valor: float) -> list[float]:
        """Retorna lista de 7 pertinências gaussianas para um valor."""
        return [_gaussiana(valor, c, self.sigma_entrada) for c in self._CENTROS]