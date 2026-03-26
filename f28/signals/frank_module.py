import numpy as np
from scipy.stats import gaussian_kde, entropy
from signals.base_signal import BaseSignal

class FrankSignalEngine(BaseSignal):
    def __init__(self, vpin_threshold: float = 0.75, entropy_limit: float = 2.0, 
                 bucket_volume: int = 1000, window_size: int = 50, 
                 steady_state_lookback: int = 5000):
        
        # Hyperparameters
        self.vpin_threshold = vpin_threshold
        self.entropy_limit = entropy_limit
        self.bucket_volume = bucket_volume 
        self.window_size = window_size
        self.steady_state_lookback = steady_state_lookback
        
        # VPIN State Tracking
        self.current_bucket_vol = 0
        self.current_buy_vol = 0
        self.current_sell_vol = 0
        self.bucket_imbalances = []  # Stores |V_buy - V_sell|
        self.last_price = None
        
        # Entropy State Tracking
        self.price_history = []
        
    def process_tick(self, timestamp, price, volume):
        """Routes raw tick data to the internal math engines."""
        self._update_vpin_state(price, volume)
        self._update_entropy_state(price)

    def is_death_signal_triggered(self) -> bool:
        """The Master Logic Gate for F-28 Phase 1."""
        current_vpin = self._calculate_vpin()
        current_entropy = self._calculate_kl_divergence()
        
        if current_vpin is None or current_entropy is None:
            return False
            
        if current_vpin > self.vpin_threshold or current_entropy > self.entropy_limit:
            return True
            
        return False

    def _update_vpin_state(self, price, volume):
        """Classifies volume and fills volume-time buckets."""
        if self.last_price is None:
            self.last_price = price
            return

        # 1. Tick Test (Lee-Ready Approximation) to classify order flow
        if price > self.last_price:
            self.current_buy_vol += volume
        elif price < self.last_price:
            self.current_sell_vol += volume
        else:
            # If price is unchanged, split volume evenly (or inherit previous tick sign)
            self.current_buy_vol += volume / 2
            self.current_sell_vol += volume / 2
            
        self.current_bucket_vol += volume
        self.last_price = price

        # 2. Check if the bucket is full
        if self.current_bucket_vol >= self.bucket_volume:
            imbalance = abs(self.current_buy_vol - self.current_sell_vol)
            self.bucket_imbalances.append(imbalance)
            
            # Maintain rolling window
            if len(self.bucket_imbalances) > self.window_size:
                self.bucket_imbalances.pop(0)
                
            # Reset bucket
            self.current_bucket_vol = 0
            self.current_buy_vol = 0
            self.current_sell_vol = 0

    def _calculate_vpin(self):
        """Calculates Volume-Synchronized Probability of Informed Trading."""
        if len(self.bucket_imbalances) < self.window_size:
            return None # Not enough data to calculate a stable VPIN
            
        # VPIN = Sum of absolute imbalances / (Number of buckets * Volume per bucket)
        total_imbalance = sum(self.bucket_imbalances)
        total_volume = self.window_size * self.bucket_volume
        
        return total_imbalance / total_volume

    # ----------------------------------------------------------------------
    # INFORMATION THEORY: KL-DIVERGENCE CALCULATION
    # ----------------------------------------------------------------------
    def _update_entropy_state(self, price):
        """Maintains the rolling price window for KDE fitting."""
        self.price_history.append(price)
        
        if len(self.price_history) > self.steady_state_lookback:
            self.price_history.pop(0)

    def _calculate_kl_divergence(self):
        """Computes the Relative Entropy between Steady State and Current State."""
        # Need enough data for the long-term baseline and the short-term comparison
        short_term_window = int(self.steady_state_lookback * 0.1) # e.g., last 10% of data
        
        if len(self.price_history) < self.steady_state_lookback:
            return None

        # 1. Calculate log returns to ensure stationarity
        prices = np.array(self.price_history)
        log_returns = np.diff(np.log(prices))
        
        # 2. Split into baseline (P) and current (Q) distributions
        steady_state_returns = log_returns[:-short_term_window]
        current_returns = log_returns[-short_term_window:]

        # 3. Fit Gaussian KDEs
        # Using Silverman's rule of thumb for bandwidth selection natively in scipy
        kde_p = gaussian_kde(steady_state_returns)
        kde_q = gaussian_kde(current_returns)

        # 4. Evaluate PDFs over a common grid space
        # We span the grid across the min and max of all observed returns
        grid_min = np.min(log_returns)
        grid_max = np.max(log_returns)
        grid = np.linspace(grid_min, grid_max, 500)

        p_pdf = kde_p.evaluate(grid)
        q_pdf = kde_q.evaluate(grid)

        # Prevent division by zero or log(0) in the entropy calculation
        p_pdf = np.where(p_pdf == 0, 1e-10, p_pdf)
        q_pdf = np.where(q_pdf == 0, 1e-10, q_pdf)

        # 5. Calculate KL Divergence: D_KL(P || Q)
        # Scipy's entropy function calculates sum(pk * log(pk / qk))
        kl_div = entropy(pk=p_pdf, qk=q_pdf)
        
        return kl_div

    def get_signal_name(self) -> str:
        return "Frank_VPIN_KDE_v1"