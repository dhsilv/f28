import numpy as np

class ExecutionEngine:
    def __init__(self, hmm_model, total_time_steps: int = 10):
        """
        hmm_model: The trained HMM instance from Phase 3.
        total_time_steps: The number of discrete execution slices (e.g., 10 one-minute buckets).
        """
        self.hmm = hmm_model
        self.T = total_time_steps
        
        # State Mapping to Urgency (kappa)
        # 0 = Quiet (Low Urgency, act like TWAP)
        # 1 = Trending (Medium Urgency, front-load slightly)
        # 2 = Distressed (High Urgency, dump risk immediately)
        self.kappa_map = {
            0: 0.1,  
            1: 1.5,  
            2: 5.0   
        }
        
        # Internal State
        self.is_executing = False
        self.target_tenor = None
        self.initial_qty = 0
        self.current_step = 0
        self.fractional_remainder = 0.0

    def initiate_roll(self, target_tenor: str, target_qty: int):
        """Triggered by the Master Strategy when Phase 1 and 2 complete."""
        self.is_executing = True
        self.target_tenor = target_tenor
        self.initial_qty = target_qty
        self.current_step = 0
        self.fractional_remainder = 0.0 # Reset on new execution
        print(f"Execution Engine: Initiating roll to {target_tenor}. Qty: {target_qty}")

    def get_next_order_size(self, current_l3_features: np.ndarray) -> int:
        """
        Calculates the exact number of contracts to execute in the current time step.
        """
        if not self.is_executing:
            return 0
            
        if self.current_step >= self.T:
            self.is_executing = False
            return 0

        # 1. Ask the HMM for the current market regime
        current_state = self.hmm.predict_online(current_l3_features)
        
        # 2. Get the corresponding urgency parameter
        kappa = self.kappa_map[current_state]

        # 3. Calculate remaining inventory for this step (t) and the next step (t+1)
        # Using the continuous-time sinh solution sampled at discrete intervals
        inventory_t = self._calculate_sinh_inventory(self.current_step, kappa)
        inventory_next = self._calculate_sinh_inventory(self.current_step + 1, kappa)

        # 4. The order size is the difference between what we hold now and what we SHOULD hold next
        theoretical_trade_size = inventory_t - inventory_next
        
        # 5. Add whatever fractional quantity we carried over from the last tick
        total_target = theoretical_trade_size + self.fractional_remainder
        
        # 6. Floor it to get the executable integer
        actual_trade = int(np.floor(total_target))
        
        # 7. Save the new fractional remainder for the next tick
        self.fractional_remainder = total_target - actual_trade
        
        self.current_step += 1
        return actual_trade

    def _calculate_sinh_inventory(self, t: int, kappa: float) -> float:
        """The mathematical core: Euler-Lagrange optimal trajectory."""
        if t >= self.T:
            return 0.0
            
        # Prevent division by zero for extremely low kappa (TWAP limit)
        if kappa < 1e-4:
            return self.initial_qty * (1.0 - (t / self.T))
            
        numerator = np.sinh(kappa * (self.T - t))
        denominator = np.sinh(kappa * self.T)
        
        return self.initial_qty * (numerator / denominator)

    def emergency_liquidate(self):
        """Triggered by the Totem Protocol (Phase 4) if the market breaks."""
        print("EXECUTION ABORT: Totem Protocol Triggered. Dumping inventory to market.")
        self.is_executing = False
        # Logic to route remaining inventory as a Market Order goes here