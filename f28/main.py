from engine.backtester import TickEngine
from strategy.f28_master import F28Strategy
from signals.frank_module import FrankSignalEngine
from alpha.curve_geometry import PCAModel
from execution.almgren_chriss import ExecutionEngine
from risk.totem_protocol import TotemCircuitBreaker

#NOTE!!! THAT THIS IS ALL FILLER CODE UNTIL I MOVE STUFF INTO SS. JUST WANTED TO HAVE A CLEAR ENTRY POINT
def main():
    frank_signal = FrankSignalEngine(vpin_threshold=0.75, kde_bandwidth='silverman')
    pca_alpha = PCAModel(ewma_span=60)
    execution = ExecutionEngine(hmm_states=3)
    totem_risk = TotemCircuitBreaker(hurst_threshold=0.1)

    strategy = F28Strategy(
        signal_engine=frank_signal,
        alpha_engine=pca_alpha,
        execution_engine=execution,
        risk_engine=totem_risk
    )

    engine = TickEngine(data_path="data/f1_f2_f3_ticks.csv")
    engine.attach_strategy(strategy)

    print("Initiating Project F-28...")
    engine.run()

if __name__ == "__main__":
    main()