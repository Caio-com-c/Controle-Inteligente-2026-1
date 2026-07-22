"""
neural_controller.py — Controlador Neural aprimorado (MLP 2 camadas)
=====================================================================

Melhorias em relação à versão anterior
---------------------------------------
1. 3 entradas em vez de 1:
   [erro, Δe, Σe_norm]  → a rede tem contexto proporcional,
   derivativo E integral — análogo a um PID mas aprendido.

2. 2 camadas ocultas (12 + 8 neurônios) em vez de 1 (6):
   mais capacidade de aproximar a política de controle não-linear
   do tanque de Torricelli.

3. Estimativa de dy/du suavizada por EMA (α=0.1):
   reduz ruído numérico na estimativa da sensibilidade da planta,
   evitando gradientes explosivos que desestabilizavam o aprendizado.

4. Gradient clipping (norma máx dos gradientes):
   impede atualizações de peso explosivas em transitórios grandes,
   que antes causavam Qe = 0 logo após o start.

5. Momentum (β=0.9) na descida do gradiente:
   acelera convergência e suaviza oscilações de aprendizado.

6. Normalização da integral do erro:
   evita que Σe exploda durante erros prolongados.

Arquitetura
-----------
    x = [e_norm, Δe_norm, Σe_norm]   (3,)
    Z1 = W1·x + b1  → A1 = tanh(Z1) (12,)
    Z2 = W2·A1 + b2 → A2 = tanh(Z2)  (8,)
    Z3 = W3·A2 + b3 → saída linear    (1,)
    Qe = clip(Z3 * Qe_max, 0, Qe_max)
"""

import numpy as np

_CLIP_GRAD = 1.0    # norma máxima dos gradientes (clipping)
_BETA      = 0.9    # momentum
_EMA_ALPHA = 0.1    # suavização da sensibilidade dy/du


class NeuroController:
    """
    Controlador neural MLP 3→12→8→1 com aprendizado online.

    Parâmetros
    ----------
    Qe_max      : float — Vazão máxima [cm³/s]; escala a saída.
    hidden1     : int   — Neurônios na 1ª camada oculta (padrão 12).
    hidden2     : int   — Neurônios na 2ª camada oculta (padrão 8).
    learn_rate  : float — Taxa de aprendizado η (padrão 0.005).
    seed        : int   — Semente de reprodutibilidade.
    """

    N_IN = 3   # erro, Δe, Σe_norm

    def __init__(
        self,
        Qe_max: float,
        hidden1: int    = 12,
        hidden2: int    = 8,
        learn_rate: float = 0.005,
        seed: int       = 42,
    ) -> None:
        self.Qe_max     = float(Qe_max)
        self.hidden1    = int(hidden1)
        self.hidden2    = int(hidden2)
        self.learn_rate = float(learn_rate)

        self._rng  = np.random.default_rng(seed)
        self._seed = seed

        self._init_pesos()

        # ── Estado do forward (reutilizado no backprop) ───────────────
        self._X  : np.ndarray | None = None
        self._Z1 : np.ndarray | None = None
        self._A1 : np.ndarray | None = None
        self._Z2 : np.ndarray | None = None
        self._A2 : np.ndarray | None = None
        self._Z3 : np.ndarray | None = None

        # ── Integral do erro ──────────────────────────────────────────
        self._integral_e : float = 0.0
        self._e_prev     : float = 0.0   # Δe = e[k] - e[k-1]

        # ── Sensibilidade da planta (dy/du) com EMA ───────────────────
        self._h_prev   : float = 0.0
        self._Qe_prev  : float = 0.0
        self._dy_du    : float = 0.01    # estimativa suavizada

        # ── Momentum (velocidades) ────────────────────────────────────
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        self._vW3 = np.zeros_like(self.W3)
        self._vb3 = np.zeros_like(self.b3)

    # ================================================================== #
    #  Forward                                                            #
    # ================================================================== #

    def forward(self, erro_pct: float, dt: float = 0.1) -> float:
        """
        Calcula o sinal de controle Qe.

        Parâmetros
        ----------
        erro_pct : erro (SP − h) em % de H_max  [−100, +100]
        dt       : passo de tempo [s] para integrar o erro

        Retorna
        -------
        float : Qe em [0, Qe_max]
        """
        # Atualiza integral (limitada)
        self._integral_e  = float(np.clip(
            self._integral_e + erro_pct * dt, -500.0, 500.0))

        # Monta vetor de entrada normalizado
        e_n   = float(np.clip(erro_pct / 100.0,           -1.0, 1.0))
        de_n  = float(np.clip((erro_pct - self._e_prev) / 100.0, -1.0, 1.0))
        ie_n  = float(np.clip(self._integral_e / 500.0,   -1.0, 1.0))
        self._e_prev = erro_pct

        self._X  = np.array([[e_n, de_n, ie_n]])    # (1, 3)

        # Camada 1
        self._Z1 = self._X  @ self.W1 + self.b1     # (1, 12)
        self._A1 = np.tanh(self._Z1)

        # Camada 2
        self._Z2 = self._A1 @ self.W2 + self.b2     # (1, 8)
        self._A2 = np.tanh(self._Z2)

        # Saída linear
        self._Z3 = self._A2 @ self.W3 + self.b3     # (1, 1)

        # Desnormaliza e satura
        Qe = float(np.clip(self._Z3[0, 0] * self.Qe_max, 0.0, self.Qe_max))
        return Qe

    # ================================================================== #
    #  Estimativa dy/du com EMA                                           #
    # ================================================================== #

    def update_sensitivity(self, h_atual: float, Qe_atual: float) -> None:
        """
        Estima dy/du = Δh/ΔQe e suaviza por EMA para reduzir ruído.
        Chamar ANTES do backprop, com os valores do passo atual.
        """
        delta_Qe = Qe_atual - self._Qe_prev
        delta_h  = h_atual  - self._h_prev

        if abs(delta_Qe) > 0.5:   # só atualiza com variação significativa
            dy_du_raw   = delta_h / delta_Qe
            # Limita para evitar sensibilidades absurdas
            dy_du_raw   = float(np.clip(dy_du_raw, -5.0, 5.0))
            # EMA: suaviza estimativa
            self._dy_du = _EMA_ALPHA * dy_du_raw + (1 - _EMA_ALPHA) * self._dy_du

        self._h_prev  = h_atual
        self._Qe_prev = Qe_atual

    # ================================================================== #
    #  Backpropagation com clipping e momentum                            #
    # ================================================================== #

    def backprop(self, erro_pct: float) -> None:
        """
        Atualiza pesos minimizando o erro de controle.
        Usa dy/du suavizada + gradient clipping + momentum.
        """
        if self._A2 is None:
            return

        e_n   = float(np.clip(erro_pct / 100.0, -1.0, 1.0))
        dy_du = self._dy_du

        # Gradiente da saída: dJ/du = -e·(dy/du)
        d_Z3 = np.array([[-e_n * dy_du]])       # (1, 1)

        # ── Camada 3 → saída ──────────────────────────────────────────
        dW3 = self._A2.T @ d_Z3                  # (8, 1)
        db3 = d_Z3.copy()

        # ── Camada 2 → oculta 2 ──────────────────────────────────────
        d_A2 = d_Z3  @ self.W3.T                 # (1, 8)
        d_Z2 = d_A2  * (1.0 - np.tanh(self._Z2) ** 2)
        dW2  = self._A1.T @ d_Z2                 # (12, 8)
        db2  = d_Z2.copy()

        # ── Camada 1 → oculta 1 ──────────────────────────────────────
        d_A1 = d_Z2  @ self.W2.T                 # (1, 12)
        d_Z1 = d_A1  * (1.0 - np.tanh(self._Z1) ** 2)
        dW1  = self._X.T @ d_Z1                  # (3, 12)
        db1  = d_Z1.copy()

        # ── Gradient clipping (norma global) ─────────────────────────
        grads  = [dW1, db1, dW2, db2, dW3, db3]
        norma  = float(np.sqrt(sum(np.sum(g**2) for g in grads)))
        if norma > _CLIP_GRAD:
            fator = _CLIP_GRAD / norma
            grads = [g * fator for g in grads]
        dW1, db1, dW2, db2, dW3, db3 = grads

        # ── Descida do gradiente com momentum ─────────────────────────
        lr = self.learn_rate

        self._vW1 = _BETA * self._vW1 + (1 - _BETA) * dW1
        self._vb1 = _BETA * self._vb1 + (1 - _BETA) * db1
        self._vW2 = _BETA * self._vW2 + (1 - _BETA) * dW2
        self._vb2 = _BETA * self._vb2 + (1 - _BETA) * db2
        self._vW3 = _BETA * self._vW3 + (1 - _BETA) * dW3
        self._vb3 = _BETA * self._vb3 + (1 - _BETA) * db3

        self.W1 -= lr * self._vW1;  self.b1 -= lr * self._vb1
        self.W2 -= lr * self._vW2;  self.b2 -= lr * self._vb2
        self.W3 -= lr * self._vW3;  self.b3 -= lr * self._vb3

    # ================================================================== #
    #  Reset / utilitários                                                #
    # ================================================================== #

    def reset(self, seed: int | None = None) -> None:
        """Reinicia pesos, momentums e estado interno."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._init_pesos()
        self._integral_e = 0.0
        self._e_prev     = 0.0
        self._h_prev     = 0.0
        self._Qe_prev    = 0.0
        self._dy_du      = 0.01
        self._X = self._Z1 = self._A1 = None
        self._Z2 = self._A2 = self._Z3 = None
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        self._vW3 = np.zeros_like(self.W3)
        self._vb3 = np.zeros_like(self.b3)

    def _init_pesos(self) -> None:
        """Inicializa pesos com Xavier e biases zero."""
        rng = self._rng
        self.W1 = rng.standard_normal((self.N_IN,    self.hidden1)) * np.sqrt(2.0 / self.N_IN)
        self.b1 = np.zeros((1, self.hidden1))
        self.W2 = rng.standard_normal((self.hidden1, self.hidden2)) * np.sqrt(2.0 / self.hidden1)
        self.b2 = np.zeros((1, self.hidden2))
        self.W3 = rng.standard_normal((self.hidden2, 1))            * np.sqrt(2.0 / self.hidden2)
        self.b3 = np.zeros((1, 1))

    def __repr__(self) -> str:
        return (
            f"NeuroController(arch={self.N_IN}→{self.hidden1}→{self.hidden2}→1, "
            f"lr={self.learn_rate}, Qe_max={self.Qe_max:.1f})"
        )