from abc import ABC, abstractmethod

class BaseSignal(ABC):
    """
    Abstract Base Class defining the mandatory interface for all Phase 1 Signal Engines.
    Any new signal model must inherit from this class and implement these exact methods.
    """

    @abstractmethod
    def process_tick(self, timestamp, price: float, volume: int):
        """
        Ingests the raw tick data to update internal state/buffers.
        Must be computationally light (O(1) or O(N)) to survive the live event loop.
        """
        pass

    @abstractmethod
    def is_death_signal_triggered(self) -> bool:
        """
        Evaluates the internal state and returns a boolean.
        True = Structural collapse of the front-month contract. Initiate Phase 2.
        False = Market is stable. Hold inventory.
        """
        pass
        
    @abstractmethod
    def get_signal_name(self) -> str:
        """
        Returns the identifier of the signal for logging and backtest reporting.
        """
        pass