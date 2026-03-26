from datetime import datetime

class F28Strategy:
    def __init__(self, frank_engine, pca_engine, ekf_overlay, execution_engine, totem_protocol, initial_f1_qty: int):
        """
        Strict Dependency Injection. The strategy doesn't know how the math works, 
        it just knows what questions to ask the engines.
        """
        self.frank = frank_engine
        self.pca = pca_engine
        self.ekf = ekf_overlay
        self.execution = execution_engine
        self.totem = totem_protocol
        
        # State Machine
        self.state = "HOLDING_F1"  # States: HOLDING_F1, ROLLING, COMPLETED, HALTED
        self.f1_inventory = initial_f1_qty
        
    def on_tick(self, tick_data: dict):
        """
        The main event loop. This is equivalent to OnQuote() in Strategy Studio.
        Expected tick_data keys: 'timestamp', 'f1_price', 'f1_vol', 'f1_spread', 
                                 'curve_prices', 'curve_spreads', 'l3_features'
        """
        # ---------------------------------------------------------
        # 1. THE PARALLEL CIRCUIT BREAKER (Phase 4)
        # ---------------------------------------------------------
        # Calculate the basis (F1 - Spot) assuming Spot is index 0 in curve_prices
        current_basis = tick_data['f1_price'] - tick_data['curve_prices'][0]
        
        if self.totem.is_market_broken(current_basis, tick_data['timestamp']):
            if self.state != "HALTED":
                self._execute_emergency_halt()
            return # Block all further routing if the physics are broken

        # ---------------------------------------------------------
        # 2. STATE MACHINE ROUTING
        # ---------------------------------------------------------
        if self.state == "HOLDING_F1":
            self._evaluate_roll_initiation(tick_data)
            
        elif self.state == "ROLLING":
            self._execute_roll_trajectory(tick_data)
            
        elif self.state == "COMPLETED":
            pass # Strategy is flat/safely rolled. Do nothing.

    def _evaluate_roll_initiation(self, tick_data):
        """
        Phases 1, 2, and 2.5: Waiting for Frank to pull the plug, then finding the target.
        """
        # 1. Update Frank's state
        self.frank.process_tick(
            timestamp=tick_data['timestamp'], 
            price=tick_data['f1_price'], 
            volume=tick_data['f1_vol']
        )
        
        # 2. Check the Death Signal
        if self.frank.is_death_signal_triggered():
            print(f"[{tick_data['timestamp']}] F-28 MASTER: Front-month collapse detected.")
            
            # 3. Fundamental Physics Check (EKF)
            is_supply_shock = self.ekf.step(
                S_t=tick_data['curve_prices'][0], 
                F_market=tick_data['f1_price'], 
                tau=30/365, # Approx 1 month to expiry !!!!!
                r=0.05      # Risk free rate approx !!!!!
            )
            
            # 4. Target Selection
            if is_supply_shock:
                print("OVERRIDE: Supply shock detected. Bypassing PCA. Defaulting to F2.")
                target_tenor, target_qty = 'F2', self.f1_inventory
            else:
                # Safe to use math. Ask PCA for the optimal target.
                pca_tick = {
                    'prices': tick_data['curve_prices'], 
                    'spreads': tick_data['curve_spreads']
                }
                target_tenor, target_qty = self.pca.get_optimal_roll_target(pca_tick, self.f1_inventory)
            
            # 5. Lock in the target and transition state
            self.execution.initiate_roll(target_tenor, target_qty)
            self.state = "ROLLING"

    def _execute_roll_trajectory(self, tick_data):
        """
        Phase 3: The roll is active. Ask the HMM/Almgren-Chriss engine how much to trade right now.
        """
        # Ask the execution engine for the required volume for this specific time slice
        trade_size = self.execution.get_next_order_size(tick_data['l3_features'])
        
        if trade_size > 0:
            # Output the order to the exchange (Equivalent to SendOrder in C++)
            self._send_order(
                venue="PRIMARY_EXCHANGE", 
                tenor=self.execution.target_tenor, 
                side="BUY", 
                qty=trade_size, 
                price=tick_data['f1_price'] # In reality, cross spread or join bid based on HMM state
            )
            
            # Track remaining inventory
            self.f1_inventory -= trade_size
            
            if self.f1_inventory <= 0:
                print(f"[{tick_data['timestamp']}] F-28 MASTER: Roll completed safely.")
                self.state = "COMPLETED"

    def _execute_emergency_halt(self):
        """Hard abort sequence triggered by the Totem."""
        self.state = "HALTED"
        self.execution.emergency_liquidate()
        print("F-28 MASTER: Strategy locked. Awaiting manual reset and regime burn-in.")

    def _send_order(self, venue: str, tenor: str, side: str, qty: int, price: float):
        """Mock method for sending the order to the exchange."""
        print(f"[ORDER OUT] {side} {qty} {tenor} @ {price:.2f} | Venue: {venue}")