import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
import ta
from torch.optim import lr_scheduler, AdamW
import concurrent.futures
import time
from yfinance import download
from requests.exceptions import RequestException
import torch.nn as nn
import torch.nn.functional as F
from pxlstm import ParallelExtendedMLSTM
from yfinance.exceptions import YFRateLimitError 



# [Previous functions: safe_fetch, fetch_chunk, get_stock_data, get_daily_data, 
#  get_hourly_data, merge_datasets, add_technical_indicators, StockDataset, 
#  FunnyMachine, prepare_data, create_sequences remain unchanged]

# config params
SEQ_LENGTH = 72  
BATCH_SIZE = 32
EPOCHS = 25
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

def safe_fetch(ticker, start, end, retries=15, delay=10):
    """
    Attempt to download data with retries and exponential backoff.
    If YFRateLimitError is encountered, wait even longer.
    """
    for attempt in range(retries):
        try:
            data = download(
                ticker,
                start=start.strftime("%Y-%m-%d"), 
                end=end.strftime("%Y-%m-%d"),
                interval='1h',
                auto_adjust=True
            )
            if data is not None and not data.empty:
                return data
        except YFRateLimitError as yfle:
            print(f"YFRateLimitError on attempt {attempt+1}: {yfle}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        # Exponential backoff
        sleep_time = delay * (attempt + 1)
        print(f"Sleeping for {sleep_time} seconds before next attempt...")
        time.sleep(sleep_time)
    return None


def fetch_chunk(ticker, start, end):
    """
    Fetch a single chunk of data using safe_fetch.
    Adds an extra delay between chunks.
    """
    print(f"Fetching data from {start.date()} to {end.date()}")
    chunk = safe_fetch(ticker, start, end, retries=5, delay=10)
    # Extra delay to reduce request frequency
    time.sleep(5)
    if chunk is not None and not chunk.empty:
        try:
            # Remove multi-index if present
            chunk.columns = chunk.columns.droplevel(1)
        except Exception:
            pass
        return chunk
    else:
        print(f"Warning: no data from {start.date()} to {end.date()}")
        return None

def get_stock_data(ticker, start_date, end_date, max_workers=3):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    delta = pd.Timedelta(days=8)
    
    date_ranges = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        date_ranges.append((current_start, current_end))
        current_start = current_end

    data_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_chunk, ticker, s, e) for s, e in date_ranges]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                data_list.append(result)
                
    if data_list:
        try:
            data = pd.concat(data_list)
            data.sort_index(inplace=True)
            return data
        except ValueError as ve:
            print("No objects to concatenate:", ve)
            return pd.DataFrame()
    else:
        print("No data was fetched!")
        return pd.DataFrame()
    
def get_daily_data(ticker, start_date='2010-01-01', end_date='2023-12-31'):
    file_path = f"{ticker}_daily_{start_date}_{end_date}.csv"
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Loaded daily data from file:", file_path)
    except FileNotFoundError:
        print("File not found, downloading daily data.")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        data.to_csv(file_path)
        print("Saved daily data to file:", file_path)
    try:
        data.columns = data.columns.droplevel(1)
    except Exception:
        pass
    data.sort_index(inplace=True)
    return data


def get_hourly_data(ticker, start_date, end_date, max_workers=3):
    file_path = f"{ticker}_hourly_{start_date}_{end_date}.csv"
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("Loaded hourly data from file:", file_path)
    except FileNotFoundError:
        print("File not found, downloading hourly data.")
        data = get_stock_data(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date), max_workers)
        data.to_csv(file_path)
        print("Saved hourly data to file:", file_path)
    return data

def merge_datasets(daily_df, hourly_df):
    # Read daily data properly, skipping metadata rows
    if 'Ticker' in daily_df.index:
        # Reset index and skip metadata rows
        daily_df = daily_df.reset_index()
        daily_df = daily_df[daily_df['Price'].str.match(r'\d{4}-\d{2}-\d{2}$', na=False)]
        daily_df = daily_df.set_index('Price')
    
    # Ensure indices are datetime
    daily_df.index = pd.to_datetime(daily_df.index)
    hourly_df.index = pd.to_datetime(hourly_df.index)
    
    # Remove timezone info if present
    if hasattr(daily_df.index, "tz") and daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_localize(None)
    if hasattr(hourly_df.index, "tz") and hourly_df.index.tz is not None:
        hourly_df.index = hourly_df.index.tz_localize(None)
    
    # Drop any duplicate indices
    daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
    hourly_df = hourly_df[~hourly_df.index.duplicated(keep='first')]
    
    # Sort both dataframes by index
    daily_df = daily_df.sort_index()
    hourly_df = hourly_df.sort_index()
    
    # Resample daily data to hourly frequency
    daily_up = daily_df.resample('1h').ffill()
    
    # Merge the datasets
    merged = pd.concat([daily_up, hourly_df])
    
    # Handle duplicates keeping the most recent data
    merged = merged[~merged.index.duplicated(keep='last')]
    
    # Final sort
    merged = merged.sort_index()
    
    return merged



def add_technical_indicators(df):
    if len(df) < 20:  
        print("Data is too short, skipping indicators.")
        return df.copy()

    # Convert string columns to numeric
    numeric_columns = ['Close', 'High', 'Low', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.copy().ffill().bfill()    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Rest of your indicators code...
    rsi_indicator = ta.momentum.RSIIndicator(close=close)
    df['RSI'] = rsi_indicator.rsi()

    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(close=close)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close)
    df['ATR'] = atr.average_true_range()

    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log1p(df['Returns'].replace([-np.inf, np.inf], np.nan))
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    df['Volume_MA'] = volume.rolling(window=20).mean()
    df['Volume_STD'] = volume.rolling(window=20).std()

    df.index = pd.to_datetime(df.index)
    df['DayOfWeek'] = df.index.dayofweek
    df['MonthOfYear'] = df.index.month

    df = df.dropna().ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    return df


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).squeeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]






###############################################################################
# xLSTM
###############################################################################



class FunnyMachine(nn.Module):
    def __init__(self, input_size, hidden_size=32, matrix_size=4, dropout=0.2):
        super().__init__()
        # Extended mLSTM (without attention) used as branch:
        self.extended_mlstm = ParallelExtendedMLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            matrix_size=matrix_size,
            num_layers=2,
            dropout=dropout,
            expansion_factor=4,     # adapt if desired
        )
        self.output_layer = nn.Linear(hidden_size * matrix_size**2, 5)

    def forward(self, x):
        x = self.extended_mlstm(x)
        x = F.gelu(x)  # Add final GELU before output
        return self.output_layer(x.mean(1))
def prepare_data(df):
    df = df.copy()
    print("Initial data size:", len(df))
    df = df.ffill().bfill()

    close = df['Close']
    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log1p(df['Returns'].replace([-np.inf, np.inf], np.nan))

    df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=close
    ).average_true_range()

    bollinger = ta.volatility.BollingerBands(close=close)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['ADX'] = ta.trend.ADXIndicator(
        high=df['High'], low=df['Low'], close=close, window=14
    ).adx()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=df['Volume']
    ).on_balance_volume()
    df['Price_SMA_10'] = close.rolling(window=10, min_periods=1).mean()
    df['Price_SMA_30'] = close.rolling(window=30, min_periods=1).mean()
    df['Price_EMA_10'] = close.ewm(span=10, adjust=False, min_periods=1).mean()
    df['Historical_Vol'] = close.rolling(window=20, min_periods=1).std() / \
                          close.rolling(window=20, min_periods=1).mean()
    df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=20, min_periods=1).std()
    df['DayOfWeek'] = df.index.dayofweek
    df['MonthOfYear'] = df.index.month
    df['Price_Change_Lag1'] = df['Returns'].shift(1)
    df['Price_Change_Lag2'] = df['Returns'].shift(2)
    print("After technical indicators:", len(df))

    df = df.dropna().ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    return df

def create_sequences(scaled_data, target_scaled, seq_length, augment=True):
    X, y = [], []
    for i in range(seq_length, len(scaled_data) - 5):
        seq = scaled_data[i - seq_length:i]
        
        if augment:
            
            if np.random.random() < 0.2:
                
                noise = np.random.normal(0, 0.002 * np.std(seq), seq.shape)
                seq = seq + noise
        
            if np.random.random() < 0.15:
                
                scale = np.random.uniform(0.98, 1.02)
                seq = seq * scale
            if np.random.random() < 0.1:  # New: time warping
                warp = np.random.uniform(0.98, 1.02, seq.shape)
                seq = seq * warp

        X.append(seq)
        y.append(target_scaled[i:i+5].flatten())
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, target_scaler, epochs=100):
    device = torch.device('cpu')  # Explicitly set to CPU for MacBook Pro Intel
    model.to(device)
    
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=1e-4,
        eps=1e-8
    )
    
    l1_lambda = 1e-5  
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        div_factor=10,
        final_div_factor=1e4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    patience = 7
    min_delta = 1e-4
    waiting = 0
    GRAD_CLIP_VALUE = 0.5

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, y_batch) + l1_lambda * l1_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_VALUE)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                all_outputs.append(outputs.cpu())
                all_targets.append(y_val.cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mape = torch.mean(torch.abs((all_targets - all_outputs) / all_targets)) * 100
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAPE: {mape:.2f}%')
        
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            waiting = 0
            print("Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mape': mape,
            }, 'prediction_MSFT.pth')
        else:
            waiting += 1
            if waiting >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

def main():
    alreadygot = False
    if (alreadygot):
        daily_data = pd.read_csv('MSFT_daily_2010-01-01_2023-12-31.csv', 
                        skiprows=[1,2],  # Skip the metadata rows
                        index_col=0,
                        parse_dates=True)
        hourly_data = pd.read_csv('MSFT_hourly_2023-02-24_2025-02-16.csv', 
                         index_col=0, 
                         parse_dates=True)
    else: 
        daily_data = pd.read_csv('MSFT_daily_2010-01-01_2023-12-31.csv', 
                        skiprows=[1,2],  # Skip the metadata rows
                        index_col=0,
                        parse_dates=True)
        hourly_data = pd.read_csv('MSFT_hourly_2023-02-24_2025-02-16.csv', 
                         index_col=0, 
                         parse_dates=True)

        numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in numeric_columns:
            daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')
            hourly_data[col] = pd.to_numeric(hourly_data[col], errors='coerce')

        df = merge_datasets(daily_data, hourly_data)
    
    df = add_technical_indicators(df)
    processed_df = prepare_data(df)
    print("After prepare_data:", len(processed_df))

    target = processed_df['Close'].values.reshape(-1, 1)
    features = processed_df.drop(columns=['Close'])
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
        print(f"Training fold {fold+1}")
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target[train_idx], target[val_idx]

        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, SEQ_LENGTH)
        print("Sequence size:", len(X_train_seq))
        print(f"Train sequences: {X_train_seq.shape}, Targets: {y_train_seq.shape}")

        train_dataset = StockDataset(X_train_seq, y_train_seq)
        val_dataset = StockDataset(X_val_seq, y_val_seq)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        model = FunnyMachine(
            input_size=X_train_seq.shape[2],
            hidden_size=32,
            matrix_size=1,  # Set to 1 to reduce memory usage for MacBook Pro with Intel CPU and GPU
            dropout=0.2
        )
        print("Model Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        train_model(model, train_loader, val_loader, target_scaler, epochs=EPOCHS)
        break

if __name__ == "__main__":
    main()



