import numpy as np
import warnings
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

class MicrostructureHMM:
    def __init__(self, n_states: int = 3, lookback_window: int = 100):
        """
        n_states: The number of hidden regimes (e.g., 3: Quiet, Trending, Distressed)
        lookback_window: How many rolling observations to use for real-time Viterbi decoding
        """
        self.n_states = n_states
        self.lookback_window = lookback_window
        
        # We use a Gaussian HMM because our observations (Imbalance, Intensity, Spread)
        # are continuous variables, not discrete tokens.
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        
        self.is_trained = False
        self.recent_observations = []

    def fit_offline(self, historical_l3_features: np.ndarray):
        """
        The Machine Learning Phase (Baum-Welch Algorithm).
        historical_l3_features shape: (n_samples, n_features)
        """
        print(f"Training HMM on {len(historical_l3_features)} L3 observations...")
        
        # Suppress deprecation warnings from hmmlearn/sklearn under the hood
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(historical_l3_features)
            
        self.is_trained = True
        print("HMM Training Complete.")
        
        # Optional: Map the arbitrary state integers (0, 1, 2) to their actual 
        # physical meanings by sorting them by their fitted variances.
        self._map_hidden_states()

    def _map_hidden_states(self):
        """
        The EM algorithm assigns states randomly. We need to identify which is which.
        Assuming State 2 (Distressed) has the highest variance in spread/intensity.
        """
        # Sum the trace of the covariance matrices for each state
        variances = [np.trace(cov) for cov in self.model.covars_]
        
        # Get the sorted indices (lowest variance to highest)
        sorted_indices = np.argsort(variances)
        
        # Create a mapping dictionary: {internal_state: logical_state}
        # 0 = Quiet, 1 = Trending, 2 = Distressed
        self.state_map = {sorted_indices[0]: 0, 
                          sorted_indices[1]: 1, 
                          sorted_indices[2]: 2}

    def predict_online(self, current_features: np.ndarray) -> int:
        """
        The Real-Time Execution Phase (Viterbi Algorithm).
        Takes the latest L3 tick, updates the rolling window, and decodes the current state.
        """
        if not self.is_trained:
            raise ValueError("HMM must be trained offline before online prediction.")

        # Update rolling observation window
        self.recent_observations.append(current_features)
        if len(self.recent_observations) > self.lookback_window:
            self.recent_observations.pop(0)

        if len(self.recent_observations) < 10:
            return 0  # Default to Quiet if we don't have enough data

        # Format for hmmlearn (requires 2D array)
        obs_matrix = np.vstack(self.recent_observations)

        # Decode the hidden state path using the Viterbi algorithm
        _, hidden_states = self.model.decode(obs_matrix)
        
        # We only care about the most recent state
        internal_current_state = hidden_states[-1]
        
        # Return the logical mapped state (0, 1, or 2)
        return self.state_map[internal_current_state]

    def init_online_state(self):
        """Called when the strategy boots up."""
        # Start with the stationary distribution (or startprob_ from training)
        self.current_probs = self.model.startprob_
        
    def predict_online_fast(self, current_features: np.ndarray) -> int:
        """
        O(1) Forward Algorithm update. Runs in microseconds.
        """
        # 1. Calculate the likelihood of the current observation under each state's Gaussian
        emission_probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            emission_probs[i] = multivariate_normal.pdf(
                current_features, 
                mean=self.model.means_[i], 
                cov=self.model.covars_[i]
            )
            
        # 2. Update rule: Emission * (Previous Probs @ Transition Matrix)
        new_probs = emission_probs * (self.current_probs @ self.model.transmat_)
        
        # 3. Normalize to prevent floating point underflow (probabilities must sum to 1)
        prob_sum = np.sum(new_probs)
        if prob_sum == 0:
            # Fallback if numerical underflow occurs in extreme anomalies
            new_probs = np.ones(self.n_states) / self.n_states
        else:
            self.current_probs = new_probs / prob_sum
            
        # 4. Return the most likely state mapped to our logical integer
        most_likely_state = np.argmax(self.current_probs)
        return self.state_map[most_likely_state]