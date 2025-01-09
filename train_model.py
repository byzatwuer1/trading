import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas_ta as ta
import joblib
import os
import glob
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Veri dizini
data_dir = "data"

def load_and_prepare_data():
    """Tüm sembol verilerini yükle ve birleştir"""
    all_data = []
    
    # data dizinindeki tüm CSV dosyalarını bul
    csv_files = glob.glob(os.path.join(data_dir, "*_data.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} directory")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        try:
            # Sembol adını dosya adından çıkar
            symbol = os.path.basename(file).replace("_data.csv", "")
            
            # CSV dosyasını oku
            df = pd.read_csv(file)
            
            # Sembol sütunu ekle
            df['symbol'] = symbol
            
            all_data.append(df)
            print(f"Loaded data for {symbol}")
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from CSV files")
    
    # Tüm verileri birleştir
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Tarihe göre sırala
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.sort_values('timestamp')
    
    return combined_data

def add_technical_indicators(df):
    """Gelişmiş teknik indikatörler"""
    try:
        # Veri tiplerini float64'e dönüştür
        numeric_columns = ['high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype('float64')

        # Trend İndikatörleri
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['EMA_200'] = ta.ema(df['close'], length=200)
        
        # Momentum İndikatörleri
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Volatilite İndikatörleri
        bb = ta.bbands(df['close'], length=20)
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Middle'] = bb['BBM_20_2.0']
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Hacim bazlı indikatörler
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        # MFI hesaplaması için verileri hazırla
        mfi_series = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['MFI'] = mfi_series.astype('float64')
        
        # Momentum ve Trend Gücü
        adx = ta.adx(df['high'], df['low'], df['close'])
        if 'ADX_14' in adx.columns:
            df['ADX'] = adx['ADX_14'].astype('float64')
        
        return df
    
    except Exception as e:
        print(f"Error in add_technical_indicators: {str(e)}")
        raise

def add_price_patterns(df):
    """Fiyat paternleri ve özel özellikler"""
    # Fiyat değişim oranları
    df['Price_Change'] = df['close'].pct_change()
    df['Volume_Change'] = df['volume'].pct_change()
    
    # Trend yönü (son n mum)
    df['Trend_3'] = df['close'].rolling(3).apply(lambda x: 1 if x.is_monotonic_increasing else (-1 if x.is_monotonic_decreasing else 0))
    df['Trend_5'] = df['close'].rolling(5).apply(lambda x: 1 if x.is_monotonic_increasing else (-1 if x.is_monotonic_decreasing else 0))
    
    # Volatilite özellikleri
    df['Daily_Range'] = (df['high'] - df['low']) / df['close']
    df['Gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    return df

def add_market_features(df):
    """Piyasa durumu özellikleri"""
    # Zaman özellikleri
    df['Hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['Day_of_Week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Hacim profili
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
    
    # Fiyat momentumu
    df['Price_MA'] = df['close'].rolling(20).mean()
    df['Price_Distance_MA'] = (df['close'] - df['Price_MA']) / df['Price_MA']
    
    return df

def prepare_features(df):
    """Tüm özellikleri hazırla"""
    df = add_technical_indicators(df)
    df = add_price_patterns(df)
    df = add_market_features(df)
    return df

def create_labels(df, lookback=1, threshold=0.001):
    """Gelişmiş etiket oluşturma"""
    future_returns = df.groupby('symbol')['close'].shift(-lookback).pct_change(lookback)
    
    # 1: Önemli yükseliş, 0: Nötr, -1: Önemli düşüş
    labels = pd.cut(future_returns, 
                   bins=[-np.inf, -threshold, threshold, np.inf],
                   labels=[-1, 0, 1])
    return labels

try:
    print("Loading data...")
    data = load_and_prepare_data()
    
    print("Preparing features...")
    data = prepare_features(data)
    
    # Özellik seçimi
    feature_columns = ['open', 'high', 'low', 'close', 'volume',
                      'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
                      'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                      'BB_Width', 'OBV', 'MFI', 'ADX',
                      'Price_Change', 'Volume_Change',
                      'Trend_3', 'Trend_5', 'Daily_Range', 'Gap',
                      'Volume_Ratio', 'Price_Distance_MA']
    
    features = data[feature_columns].copy()
    labels = create_labels(data, lookback=1, threshold=0.001)
    
    # NaN değerleri temizle
    features = features.fillna(method='ffill').fillna(0)
    valid_idx = ~labels.isna()
    features = features[valid_idx]
    labels = labels[valid_idx]
    
    # Veriyi ölçekle
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Zaman serisi cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model parametreleri
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # Model eğitimi ve değerlendirme
    scores = []
    for train_idx, test_idx in tscv.split(scaled_features):
        X_train = scaled_features[train_idx]
        X_test = scaled_features[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Performans metrikleri
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        scores.append({'precision': precision, 'recall': recall, 'f1': f1})
    
    # Ortalama performans
    avg_scores = pd.DataFrame(scores).mean()
    print("\nModel Performance:")
    print(f"Precision: {avg_scores['precision']:.4f}")
    print(f"Recall: {avg_scores['recall']:.4f}")
    print(f"F1 Score: {avg_scores['f1']:.4f}")
    
    # Özellik önemlilik analizi
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    # Model ve scaler'ı kaydet
    joblib.dump(model, 'models/ml_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\nModel and scaler successfully saved.")

except Exception as e:
    print(f"Error during training: {str(e)}")
    raise
