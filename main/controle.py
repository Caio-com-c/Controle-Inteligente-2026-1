import time
import threading
import queue
import numpy as np


class Control:
    """
    Simulação em tempo real do tanque de nível (Torricelli + Euler).

    Modos de controle
    -----------------
    "none"         — malha aberta, Qe fixa
    "onoff"        — liga/desliga com histerese configurável
    "pid"          — PID discreto posicional com anti-windup
    "pid_adaptive" — PID com ganhos variáveis por zona de erro

    Ruído (distúrbio de sensor)
    ---------------------------
    Gaussiano de média zero aplicado APÓS o cálculo físico,
    simulando erro de leitura do sensor:
        h_medido  = h_real  + N(0, σ_h)   se ruido_y ativo
        Qe_medido = Qe_real + N(0, σ_Qe)  se ruido_u ativo
    σ é fixo em 1 % de H_max (h) e 1 % de Qe_nominal (Qe).

    Filtro passa-baixa (EMA)
    ------------------------
    y_f[k] = alpha * y[k] + (1 - alpha) * y_f[k-1]
        alpha ∈ (0, 1]: 1 = sem filtragem, ~0 = muito suave
    Aplicado sobre o sinal já ruidoso, separadamente para h e Qe.

    Estado publicado em get_state()
    --------------------------------
    sim_time   — tempo simulado [s]
    h_raw      — altura real (sem ruído/filtro) [%]
    h_pct      — altura com ruído + filtro [%]   ← o que a interface exibe
    sp_pct     — setpoint [%]
    error_pct  — (sp - h_pct) em % de H_max
    Qe_raw     — Qe real [cm³/s]
    Qe         — Qe com ruído + filtro [cm³/s]
    Qs         — vazão de saída Torricelli [cm³/s]
    u_ctrl     — sinal de controle antes da saturação
    Kp/Ki/Kd   — ganhos ativos
    mode       — modo ativo
    """

    def __init__(
        self,
        A_t: float,
        A_f: float,
        H_max: float,
        Qe_nominal: float,
        g: float = 981.0,
        dt: float = 0.1,
        speed: float = 1.0,
    ) -> None:
        if A_t <= 0 or A_f <= 0:
            raise ValueError("A_t e A_f devem ser positivos.")
        if H_max <= 0:
            raise ValueError("H_max deve ser positivo.")
        if Qe_nominal <= 0:
            raise ValueError("Qe_nominal deve ser positivo.")
        if dt <= 0:
            raise ValueError("dt deve ser positivo.")

        self.A_t        = A_t
        self.A_f        = A_f
        self.H_max      = H_max
        self.Qe_nominal = Qe_nominal
        self.g          = g
        self.dt         = dt
        self.speed      = speed

        # ---- Estado da simulação ------------------------------------ #
        self._h        = 0.0
        self._sim_time = 0.0

        # ---- Lock --------------------------------------------------- #
        self._lock = threading.Lock()

        # ---- Parâmetros de controle --------------------------------- #
        self._setpoint = 0.0
        self._mode     = "none"
        self._Qe_min   = 0.0
        self._Qe_max   = Qe_nominal
        self._Qe_fixed = 0.0

        # ON/OFF
        self._onoff_histerese = 2.0   # banda em % de H_max

        # PID
        self._Kp = 0.0
        self._Ki = 0.0
        self._Kd = 0.0
        self._integral = 0.0
        self._e_prev   = 0.0

        # PID adaptativo
        self._zone_near  = 0.05
        self._zone_mid   = 0.20
        self._gains_near = (2.0,  0.20, 0.5)
        self._gains_mid  = (5.0,  0.10, 1.0)
        self._gains_far  = (10.0, 0.05, 2.0)

        # ---- Ruído -------------------------------------------------- #
        self._ruido_h  = False   # distúrbio no sensor de nível
        self._ruido_Qe = False   # distúrbio no sensor de vazão
        # σ fixo: 1 % de H_max e 1 % de Qe_nominal
        self._sigma_h  = 0.01 * H_max
        self._sigma_Qe = 0.01 * Qe_nominal

        # ---- Filtro passa-baixa (EMA) ------------------------------- #
        self._filtro_h  = False
        self._filtro_Qe = False
        self._alpha_h   = 0.3    # padrão suave
        self._alpha_Qe  = 0.3
        # Valores filtrados (estado do filtro entre iterações)
        self._h_filt   = 0.0
        self._Qe_filt  = 0.0

        # ---- Fila de saída ------------------------------------------ #
        self._state_queue: queue.Queue = queue.Queue(maxsize=1)

        # ---- Controle da thread ------------------------------------- #
        self._running = False
        self._thread: threading.Thread | None = None

    # ================================================================== #
    #  API pública                                                        #
    # ================================================================== #

    def start(self, h0: float = 0.0) -> None:
        """Inicia (ou reinicia) a simulação do zero."""
        self.stop()
        with self._lock:
            self._h        = float(h0)
            self._sim_time = 0.0
            self._integral = 0.0
            self._e_prev   = 0.0
            self._h_filt   = float(h0)
            self._Qe_filt  = self._Qe_fixed
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def pause(self) -> None:
        self._running = False

    def resume(self) -> None:
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def reset(self, h0: float = 0.0) -> None:
        self.start(h0=h0)

    # ------------------------------------------------------------------ #
    #  Setters — controle                                                 #
    # ------------------------------------------------------------------ #

    def set_setpoint(self, value_pct: float) -> None:
        """Setpoint em % de H_max. Reseta o integrador."""
        with self._lock:
            self._setpoint = float(value_pct) / 100.0 * self.H_max
            self._integral = 0.0
            self._e_prev   = 0.0

    def set_mode(self, mode: str) -> None:
        """'none' | 'onoff' | 'pid' | 'pid_adaptive'"""
        with self._lock:
            self._mode     = mode
            self._integral = 0.0
            self._e_prev   = 0.0

    def set_pid_gains(self, Kp: float, Ki: float, Kd: float) -> None:
        with self._lock:
            self._Kp = float(Kp)
            self._Ki = float(Ki)
            self._Kd = float(Kd)

    def set_Qe_fixed(self, Qe: float) -> None:
        with self._lock:
            self._Qe_fixed = float(np.clip(Qe, 0.0, self.Qe_nominal))

    def set_onoff_histerese(self, banda_pct: float) -> None:
        with self._lock:
            self._onoff_histerese = float(banda_pct)

    def set_adaptive_zones(self, zone_near, zone_mid,
                           gains_near, gains_mid, gains_far) -> None:
        with self._lock:
            self._zone_near  = zone_near
            self._zone_mid   = zone_mid
            self._gains_near = tuple(gains_near)
            self._gains_mid  = tuple(gains_mid)
            self._gains_far  = tuple(gains_far)

    # ------------------------------------------------------------------ #
    #  Setters — ruído                                                    #
    # ------------------------------------------------------------------ #

    def set_ruido(self, h: bool = False, Qe: bool = False) -> None:
        """
        Ativa/desativa distúrbio gaussiano nos sensores.
        h=True  → ruído no sensor de nível
        Qe=True → ruído no sensor de vazão de entrada
        """
        with self._lock:
            self._ruido_h  = bool(h)
            self._ruido_Qe = bool(Qe)

    # ------------------------------------------------------------------ #
    #  Setters — filtro                                                   #
    # ------------------------------------------------------------------ #

    def set_filtro(self, h: bool = False, Qe: bool = False,
                   alpha_h: float = 0.3, alpha_Qe: float = 0.3) -> None:
        """
        Ativa/desativa o filtro EMA e define os alphas.
        alpha ∈ (0, 1]: 1.0 = sem efeito, ~0.05 = muito suave.
        """
        alpha_h  = float(np.clip(alpha_h,  1e-3, 1.0))
        alpha_Qe = float(np.clip(alpha_Qe, 1e-3, 1.0))
        with self._lock:
            self._filtro_h   = bool(h)
            self._filtro_Qe  = bool(Qe)
            self._alpha_h    = alpha_h
            self._alpha_Qe   = alpha_Qe

    # ------------------------------------------------------------------ #
    #  Getter                                                             #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict | None:
        try:
            return self._state_queue.get_nowait()
        except queue.Empty:
            return None

    # ================================================================== #
    #  Loop da thread                                                     #
    # ================================================================== #

    def _loop(self) -> None:
        onoff_ligado = False

        while self._running:
            t_start = time.perf_counter()

            # --- Snapshot atômico ------------------------------------ #
            with self._lock:
                h         = self._h
                setpoint  = self._setpoint
                mode      = self._mode
                Kp        = self._Kp
                Ki        = self._Ki
                Kd        = self._Kd
                Qe_fixed  = self._Qe_fixed
                Qe_min    = self._Qe_min
                Qe_max    = self._Qe_max
                integral  = self._integral
                e_prev    = self._e_prev
                histerese = self._onoff_histerese / 100.0 * self.H_max
                zone_near = self._zone_near
                zone_mid  = self._zone_mid
                g_near    = self._gains_near
                g_mid     = self._gains_mid
                g_far     = self._gains_far
                ruido_h   = self._ruido_h
                ruido_Qe  = self._ruido_Qe
                filtro_h  = self._filtro_h
                filtro_Qe = self._filtro_Qe
                alpha_h   = self._alpha_h
                alpha_Qe  = self._alpha_Qe
                h_filt    = self._h_filt
                Qe_filt   = self._Qe_filt

            # --- Controlador usa h REAL (sem ruído) ------------------ #
            u_ctrl = 0.0
            active_Kp, active_Ki, active_Kd = Kp, Ki, Kd

            if mode == "none":
                Qe = Qe_fixed
                e  = setpoint - h

            elif mode == "onoff":
                e = setpoint - h
                dead_lo = setpoint - histerese / 2.0
                dead_hi = setpoint + histerese / 2.0
                if h <= dead_lo:
                    onoff_ligado = True
                elif h >= dead_hi:
                    onoff_ligado = False
                Qe     = Qe_max if onoff_ligado else Qe_min
                u_ctrl = Qe

            else:
                e = setpoint - h

                if mode == "pid_adaptive":
                    rel_err = abs(e / setpoint) if setpoint != 0 else abs(e) / self.H_max
                    if rel_err > zone_mid:
                        active_Kp, active_Ki, active_Kd = g_far
                    elif rel_err > zone_near:
                        active_Kp, active_Ki, active_Kd = g_mid
                    else:
                        active_Kp, active_Ki, active_Kd = g_near

                P = active_Kp * e
                D = active_Kd * (e - e_prev) / self.dt
                integral += e * self.dt
                I = active_Ki * integral
                u_ctrl = P + I + D
                Qe = float(np.clip(u_ctrl, Qe_min, Qe_max))
                if u_ctrl != Qe:
                    integral -= e * self.dt

            # --- Torricelli + Euler (física real) -------------------- #
            h = max(h, 0.0)
            Qs     = self.A_f * np.sqrt(2.0 * self.g * h) if h > 0 else 0.0
            dh     = (Qe - Qs) / self.A_t
            h_next = float(np.clip(h + dh * self.dt, 0.0, self.H_max))

            # --- Ruído: distúrbio gaussiano nos sensores ------------- #
            h_obs  = h_next
            Qe_obs = Qe

            if ruido_h:
                h_obs  = h_obs  + np.random.normal(0.0, self._sigma_h)
                h_obs  = float(np.clip(h_obs, 0.0, self.H_max))

            if ruido_Qe:
                Qe_obs = Qe_obs + np.random.normal(0.0, self._sigma_Qe)
                Qe_obs = float(np.clip(Qe_obs, 0.0, self.Qe_nominal))

            # --- Filtro EMA ------------------------------------------ #
            if filtro_h:
                h_filt = alpha_h * h_obs + (1.0 - alpha_h) * h_filt
                h_obs  = h_filt

            if filtro_Qe:
                Qe_filt = alpha_Qe * Qe_obs + (1.0 - alpha_Qe) * Qe_filt
                Qe_obs  = Qe_filt

            # --- Salva estado ---------------------------------------- #
            with self._lock:
                self._h        = h_next
                self._sim_time += self.dt
                self._integral = integral
                self._e_prev   = e
                self._h_filt   = h_filt
                self._Qe_filt  = Qe_filt

            # --- Publica --------------------------------------------- #
            sp_pct    = (setpoint / self.H_max) * 100.0
            h_obs_pct = (h_obs    / self.H_max) * 100.0
            error_pct = sp_pct - h_obs_pct

            state = {
                "sim_time":  self._sim_time,
                "h_raw":     (h_next / self.H_max) * 100.0,  # sem ruído/filtro
                "h_pct":     h_obs_pct,                       # com ruído/filtro
                "sp_pct":    sp_pct,
                "error_pct": error_pct,
                "Qe_raw":    Qe,
                "Qe":        Qe_obs,
                "Qs":        Qs,
                "u_ctrl":    u_ctrl,
                "Kp":        active_Kp,
                "Ki":        active_Ki,
                "Kd":        active_Kd,
                "mode":      mode,
            }

            try:
                self._state_queue.put_nowait(state)
            except queue.Full:
                try:
                    self._state_queue.get_nowait()
                except queue.Empty:
                    pass
                self._state_queue.put_nowait(state)

            # --- Cadência -------------------------------------------- #
            elapsed    = time.perf_counter() - t_start
            sleep_time = (self.dt / self.speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ================================================================== #
    #  Utilitários                                                        #
    # ================================================================== #

    def equilibrium(self, Qe: float) -> float:
        return (Qe / self.A_f) ** 2 / (2.0 * self.g)

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return (
            f"Control(A_t={self.A_t:.2f} cm², A_f={self.A_f:.4f} cm², "
            f"H_max={self.H_max:.2f} cm, Qe_max={self.Qe_nominal:.2f} cm³/s, "
            f"dt={self.dt} s, speed={self.speed}×)"
        )