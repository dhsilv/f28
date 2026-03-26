import os
import json
import pandas as pd
import numpy as np
import warnings
from scipy.stats import gaussian_kde
from hmmlearn.hmm import GaussianHMM

class StrategyStudioETL:
    def __init__(self, data_directory: str):
        self.data_dir = data_directory
        # If i remember correctly
        self.columns = [
            'Symbol', 'Date', 'Time', 'MsgType', 
            'Price1', 'Size1', 'Price2', 'Size2', 'Extra1', 'Extra2' #forgot the extra ones
        ]

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads all txt files in the directory and formats the timestamps."""
        print("ETL: Loading Strategy Studio text ticks...")
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        df_list = []
        for file in all_files:
            # Read CSV, ignoring malformed rows often found in raw PCAP dumps
            df = pd.read_csv(file, names=self.columns, header=None, on_bad_lines='skip')
            df_list.append(df)
            
        master_df = pd.concat(df_list, ignore_index=True)
        
        # Create a proper datetime index for resampling
        master_df['Datetime'] = pd.to_datetime(master_df['Date'].astype(str) + ' ' + master_df['Time'])
        master_df.set_index('Datetime', inplace=True)
        master_df.sort_index(inplace=True)
        
        return master_df

    def extract_frank_baseline(self, df: pd.DataFrame) -> dict:
        """Phase 1: Extracts Trade (T) prices and calculates the KDE baseline."""
        print("ETL: Calculating Frank's Steady-State KDE...")
        
        # Filter only for Trades
        trades = df[df['MsgType'] == 'T'].copy()
        
        # Assuming Price1 is the execution price
        trades['LogReturn'] = np.log(trades['Price1'] / trades['Price1'].shift(1))
        trades.dropna(inplace=True)
        
        # Fit the KDE
        returns_array = trades['LogReturn'].values
        kde = gaussian_kde(returns_array, bw_method='silverman')
        
        # We can't serialize a scipy object to JSON natively, 
        # so we evaluate it over a grid and save the probability map.
        grid = np.linspace(np.min(returns_array), np.max(returns_array), 500)
        pdf = kde.evaluate(grid)
        
        return {
            "grid": grid.tolist(),
            "pdf": pdf.tolist()
        }

    def extract_hmm_features(self, df: pd.DataFrame, bucket_freq='1S') -> np.ndarray:
        """Phase 3: Merges Quotes and Trades to build the L3 feature matrix."""
        print("ETL: Building L3 Feature Matrix for HMM...")
        
        # 1. Isolate Quotes (Q) for Spread and Imbalance
        quotes = df[df['MsgType'] == 'Q'].copy()
        # Strategy Studio Q format: Price1=Bid, Size1=BidSize, Price2=Ask, Size2=AskSize
        quotes['Spread'] = quotes['Price2'] - quotes['Price1']
        quotes['Imbalance'] = (quotes['Size1'] - quotes['Size2']) / (quotes['Size1'] + quotes['Size2'])
        
        # Resample quotes to 1-second buckets (taking the mean state of the book)
        q_resampled = quotes[['Spread', 'Imbalance']].resample(bucket_freq).mean().ffill()
        
        # 2. Isolate Trades (T) for Intensity
        trades = df[df['MsgType'] == 'T'].copy()
        trades['Volume'] = trades['Size1']
        
        # Resample trades to sum volume per second
        t_resampled = trades[['Volume']].resample(bucket_freq).sum().fillna(0)
        
        # 3. Merge features
        features = pd.concat([q_resampled, t_resampled], axis=1).dropna()
        
        # Convert to numpy matrix [Spread, Imbalance, Intensity]
        return features.values

    def train_hmm(self, feature_matrix: np.ndarray) -> dict:
        """Trains the Baum-Welch algorithm and extracts the matrices."""
        print(f"ETL: Training HMM on {len(feature_matrix)} observations...")
        
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(feature_matrix)
            
        # Extract parameters for C++ / Strategy Studio consumption
        return {
            "transition_matrix": model.transmat_.tolist(),
            "means": model.means_.tolist(),
            "covars": model.covars_.tolist(),
            "start_prob": model.startprob_.tolist()
        }

# --- Execution Execution ---
if __name__ == "__main__":
    etl = StrategyStudioETL(data_directory="./data/feb_to_apr_ticks")
    
    # 1. Parse Data
    master_data = etl.load_and_clean_data()
    
    # 2. Extract Models
    frank_data = etl.extract_frank_baseline(master_data)
    
    hmm_features = etl.extract_hmm_features(master_data)
    hmm_data = etl.train_hmm(hmm_features)
    
    # 3. Serialize to JSON for Strategy Studio
    output = {
        "frank_kde": frank_data,
        "hmm_matrices": hmm_data
    }
    
    with open("./models/f28_parameters.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("ETL COMPLETE: f28_parameters.json successfully written.")