import numpy as np

class PCAModel:
    def __init__(self, num_tenors: int = 5, ewma_span: int = 60, 
                 num_components: int = 3, liquidity_penalty_bps: float = 2.0):
        
        self.num_tenors = num_tenors
        self.num_components = num_components # Usually 3: Level, Slope, Curvature
        self.liquidity_penalty = liquidity_penalty_bps / 10000.0
        
        # EWMA Decay Factor (Lambda)
        self.decay = (ewma_span - 1) / (ewma_span + 1)
        
        # State Matrices
        self.ewma_mean = np.zeros(num_tenors)
        self.ewma_cov = np.eye(num_tenors) # Initialize as Identity matrix
        self.eigenvectors = np.eye(num_tenors)
        
        # Curve History
        self.last_prices = None
        self.last_symbol = None # Tracks the exact physical contract
        self.is_initialized = False

    def _update_ewma_covariance(self, current_prices: np.ndarray):
        """Updates covariance, protected by the Roll Gap Stitch.""" 
        if self.last_prices is None:
            self.last_prices = current_prices
            self.last_symbol = current_f1_symbol
            return

        # THE STITCH: If the symbol rolled overnight, do not calculate a return!
        if current_f1_symbol != self.last_symbol:
            print(f"ROLL GAP DETECTED: {self.last_symbol} -> {current_f1_symbol}. Stitching data.")
            self.last_prices = current_prices # Reset baseline to new contract prices
            self.last_symbol = current_f1_symbol
            return # Skip the EWMA math for this single tick to avoid corruption

        # 1. Safe to calculate log returns
        returns = np.log(current_prices / self.last_prices)
        self.last_prices = current_prices
        if not self.is_initialized:
            self.ewma_mean = returns
            # Needs a burn-in period in live trading, but we initialize to start
            self.is_initialized = True
            return

        # 2. Update EWMA Mean
        self.ewma_mean = self.decay * self.ewma_mean + (1 - self.decay) * returns
        
        # 3. Mean-centered returns
        centered_returns = returns - self.ewma_mean
        
        # 4. Update EWMA Covariance Matrix
        # Formula: Sigma_t = lambda * Sigma_{t-1} + (1-lambda) * (R_t * R_t^T)
        return_matrix = centered_returns.reshape(-1, 1)
        instantaneous_cov = return_matrix @ return_matrix.T
        
        self.ewma_cov = self.decay * self.ewma_cov + (1 - self.decay) * instantaneous_cov

    def _perform_pca(self):
        """Extracts orthogonal risk factors using Eigen-decomposition."""
        if not self.is_initialized:
            return

        # eigh is highly optimized for symmetric positive semi-definite matrices
        eigenvalues, eigenvectors = np.linalg.eigh(self.ewma_cov)
        
        # eigh sorts in ascending order. We need descending (largest variance first)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, idx]

    def _calculate_residuals(self, current_prices: np.ndarray) -> np.ndarray:
        """Calculates how 'cheap' or 'rich' each tenor is relative to the PCA model."""
        if self.last_prices is None or not self.is_initialized:
            return np.zeros(self.num_tenors)

        returns = np.log(current_prices / self.last_prices)
        centered_returns = returns - self.ewma_mean

        # 1. Isolate the top N Principal Components (Level, Slope, Curvature)
        P_k = self.eigenvectors[:, :self.num_components]
        
        # 2. Project returns into PCA space (Calculate factor scores)
        factor_scores = P_k.T @ centered_returns
        
        # 3. Reconstruct the theoretical returns from the factors
        theoretical_returns = P_k @ factor_scores
        
        # 4. Calculate the error (Residual epsilon)
        # Positive epsilon means market price moved higher than theoretical (Rich -> Sell)
        # Negative epsilon means market price moved lower than theoretical (Cheap -> Buy)
        residuals = centered_returns - theoretical_returns
        
        return residuals

    def get_optimal_roll_target(self, tick_data, current_f1_qty: int):
        """
        The routing logic. Compares F2 vs F3 to find the mathematically optimal roll.
        tick_data must contain current_prices array and bid_ask_spreads array.
        """
        current_prices = tick_data['prices']       # [F1, F2, F3, F4, F5]
        bid_ask_spreads = tick_data['spreads']     # [S1, S2, S3, S4, S5]
        
        # 1. Update the math
        self._update_ewma_covariance(current_prices)
        self._perform_pca()
        
        # 2. Get the raw PCA residuals
        residuals = self._calculate_residuals(current_prices)
        
        # We are looking to BUY the back month to replace our F1 inventory.
        # Therefore, we want the most NEGATIVE residual (cheapest tenor).
        raw_edge_f2 = -residuals[1]  # Index 1 is F2
        raw_edge_f3 = -residuals[2]  # Index 2 is F3
        
        # 3. Apply the Liquidity Penalty
        # Subtract the cost of crossing the spread from our theoretical edge
        tradeable_edge_f2 = raw_edge_f2 - (bid_ask_spreads[1] / current_prices[1]) - self.liquidity_penalty
        tradeable_edge_f3 = raw_edge_f3 - (bid_ask_spreads[2] / current_prices[2]) - self.liquidity_penalty

        # 4. The Skip-Roll Decision
        # If F3 edge is strictly greater than F2 edge after accounting for liquidity, we skip.
        if tradeable_edge_f3 > tradeable_edge_f2 and tradeable_edge_f3 > 0:
            target_tenor = 'F3'
            # (Optional: Adjust qty based on duration/beta matching, but 1:1 for simplicity now)
            target_qty = current_f1_qty 
        else:
            target_tenor = 'F2'
            target_qty = current_f1_qty
            
        return target_tenor, target_qty