import numpy as np
import warnings

class TotemCircuitBreaker:
    def __init__(self, window_size: int = 100, jump_threshold: float = 2.5, hurst_limit: float = 0.15):
        """
        window_size: Number of ticks to calculate rolling variations.
        jump_threshold: How much larger RV must be than BPV to flag a structural jump.
        hurst_limit: The floor for the Hurst Exponent. Below this, the market is deterministically pinned.
        """
        self.window_size = window_size
        self.jump_threshold = jump_threshold
        self.hurst_limit = hurst_limit
        
        self.basis_history = []
        self.log_returns = []
        
        self.is_halted = False

    def is_market_broken(self, current_basis: float, timestamp) -> bool:
        """
        The continuous physics check. Returns True if the market laws have failed.
        """
        if self.is_halted:
            return True # Lock out the system until manual reset

        self.basis_history.append(current_basis)
        
        if len(self.basis_history) > 1:
            # Calculate tick-to-tick log return of the basis
            ret = np.log(self.basis_history[-1] / self.basis_history[-2])
            self.log_returns.append(ret)
            
        if len(self.log_returns) > self.window_size:
            self.basis_history.pop(0)
            self.log_returns.pop(0)
        else:
            return False # Not enough data to assess physics

        # 1. Check for Structural Jumps (RV vs BPV)
        rv, bpv = self._calculate_variations()
        
        # If RV is massively larger than BPV, the continuous diffusion assumption is dead.
        if bpv > 0 and (rv / bpv) > self.jump_threshold:
            print(f"[{timestamp}] TOTEM PROTOCOL: Structural Jump Detected. RV/BPV ratio critical.")
            self._trigger_halt()
            return True

        # 2. Check for Market Pinning (Hurst Exponent)
        hurst = self._calculate_hurst_approximation()
        
        if hurst < self.hurst_limit:
            print(f"[{timestamp}] TOTEM PROTOCOL: The Totem is wobbling. Hurst = {hurst:.3f}.")
            print("Market is deterministically pinned. Price discovery has failed.")
            self._trigger_halt()
            return True

        return False

    def _trigger_halt(self):
        """Locks the system and issues the retraining directive."""
        self.is_halted = True
        print(">>> SYSTEM HALTED. EMERGENCY LIQUIDATION REQUIRED. <<<")
        print(">>> REGIME BREAK DETECTED. DO NOT USE PRE-BREAK DATA FOR RETRAINING. <<<")
        print(">>> Await new regime burn-in before recalculating HMM and KDE matrices. <<<")

    def _calculate_variations(self):
        """Calculates Realized Variance (RV) and Bipower Variation (BPV)."""
        returns_array = np.array(self.log_returns)
        
        # Realized Variance: Sum of squared returns
        rv = np.sum(returns_array**2)
        
        # Bipower Variation: Isolates the continuous diffusion component
        # Formula: (pi/2) * sum(|r_i| * |r_{i-1}|)
        abs_returns = np.abs(returns_array)
        bpv = (np.pi / 2.0) * np.sum(abs_returns[1:] * abs_returns[:-1])
        
        return rv, bpv

    def _calculate_hurst_approximation(self) -> float:
        """
        Calculates a fast, localized approximation of the Hurst Exponent 
        using variance scaling to avoid slowing down the OnTick loop.
        """
        returns_array = np.array(self.log_returns)
        
        # Variance of 1-tick returns
        var_1 = np.var(returns_array)
        if var_1 == 0:
            return 0.0
            
        # Variance of 2-tick aggregated returns
        returns_2 = returns_array[1:] + returns_array[:-1]
        var_2 = np.var(returns_2)
        
        # Hurst approximation derived from variance scaling: Var(k*t) = k^(2H) * Var(t)
        # Therefore: 2H = log2(var_2 / var_1)
        ratio = var_2 / var_1
        
        if ratio <= 0:
            return 0.0
            
        h = 0.5 * np.log2(ratio)
        return max(0.0, min(1.0, h)) # Bound between 0 and 1