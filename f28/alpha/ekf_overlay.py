import numpy as np

class ConvenienceYieldEKF:
    def __init__(self, dt: float, kappa: float, theta: float, 
                 sigma_y: float, obs_noise: float, physical_limit: float = 0.15):
        """
        dt: Time step (e.g., 1/252 for daily, much smaller for tick)
        kappa: Mean-reversion speed of the convenience yield
        theta: Long-term mean of the convenience yield
        sigma_y: Volatility of the convenience yield process
        obs_noise: Variance of market microstructure noise (bid-ask bounce)
        physical_limit: The threshold where y implies a physical supply shock
        """
        self.dt = dt
        self.kappa = kappa
        self.theta = theta
        self.physical_limit = physical_limit
        
        # State Initialization: [y_t]
        self.y = theta 
        self.P = np.eye(1)  # Initial uncertainty (Covariance matrix)
        
        # Noise Matrices
        self.Q = np.array([[sigma_y**2 * dt]])  # Process noise (OU variance)
        self.R = np.array([[obs_noise**2]])     # Observation noise

    def step(self, S_t: float, F_market: float, tau: float, r: float) -> bool:
        """
        Runs one cycle of the EKF. Returns True if the market is physically broken.
        """
        self._predict()
        self._update(S_t, F_market, tau, r)
        
        # The Circuit Breaker Check
        # If the convenience yield exceeds our physical limit (e.g., 15%), 
        # a supply shock is occurring. PCA signals are invalid.
        return self.y > self.physical_limit

    def _predict(self):
        """Step 1: Predict the next state based on the OU process physics."""
        # Non-linear state transition f(x)
        self.y = self.y + self.kappa * (self.theta - self.y) * self.dt
        
        # Jacobian (F) of the state transition function wrt y
        F_jacob = np.array([[1 - self.kappa * self.dt]])
        
        # Predict Error Covariance
        self.P = F_jacob @ self.P @ F_jacob.T + self.Q

    def _update(self, S_t: float, F_market: float, tau: float, r: float):
        """Step 2: Correct the estimate using the actual market observation."""
        # 1. Theoretical Observation h(y)
        F_theo = S_t * np.exp((r - self.y) * tau)
        
        # 2. Jacobian (H) of the observation function wrt y
        # dF/dy = S * exp((r-y)*tau) * (-tau) = -tau * F_theo
        H_jacob = np.array([[-tau * F_theo]])
        
        # 3. Innovation (Measurement Residual)
        residual = F_market - F_theo
        
        # 4. Innovation Covariance (S)
        S_cov = H_jacob @ self.P @ H_jacob.T + self.R
        
        # 5. Optimal Kalman Gain (K)
        # Represents how much we trust the new market data vs our internal physics model
        K = self.P @ H_jacob.T @ np.linalg.inv(S_cov)
        
        # 6. Update State and Covariance
        self.y = self.y + (K @ np.array([[residual]]))[0, 0]
        self.P = (np.eye(1) - K @ H_jacob) @ self.P

    def get_current_yield(self) -> float:
        return self.y