import asyncio
import json
import logging
from datetime import datetime, time
import time as time_module
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.um_futures import UMFutures
from binance.error import ClientError
from telegram import Bot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class BinanceFuturesBot:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.client = UMFutures(
            key=self.config['api_key'],
            secret=self.config['api_secret']
        )
        self.telegram = Bot(token=self.config['telegram_token'])
        self.chat_id = self.config['telegram_chat_id']
        self.positions = {}
        self.last_api_call = 0
        self.rate_limit_delay = 0.1
        self.model = self._load_ml_model()
        self.scaler = self._load_scaler()
        self.daily_trades = 0
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'losses': 0.0
        }
        self.last_daily_reset = datetime.now().date()

    def _load_config(self, config_path: str) -> dict:
        """Config dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            self._validate_config(config)
            return config
        except Exception as e:
            logging.error(f"Config yükleme hatası: {e}")
            raise

    def _validate_config(self, config: dict) -> None:
        """Config dosyasını doğrula"""
        required_fields = [
            'api_key', 'api_secret', 'telegram_token', 'telegram_chat_id',
            'symbols', 'risk_management', 'trading_hours', 'timeframes',
            'ml_model_path', 'scaler_path'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Eksik config alanı: {field}")

    def _load_ml_model(self) -> GradientBoostingClassifier:
        """Makine öğrenimi modelini yükle"""
        try:
            model = joblib.load(self.config['ml_model_path'])
            return model
        except Exception as e:
            logging.error(f"ML model yükleme hatası: {e}")
            raise

    def _load_scaler(self) -> StandardScaler:
        """Ölçekleyiciyi yükle"""
        try:
            scaler = joblib.load(self.config['scaler_path'])
            return scaler
        except Exception as e:
            logging.error(f"Scaler yükleme hatası: {e}")
            raise

    async def send_telegram(self, message: str) -> None:
        """Telegram mesajı gönder"""
        if self.config['notifications']['trade_updates']:
            try:
                await self.telegram.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
            except Exception as e:
                logging.error(f"Telegram mesaj hatası: {e}")

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Mum verilerini al"""
        try:
            timeframe = self.config['timeframes']['default']
            klines = self.client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=100
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logging.error(f"Kline veri alma hatası: {e}")
            return pd.DataFrame()

    def calculate_bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
            """Yutan Ayı (Bearish Engulfing) formasyonunu hesapla"""
            try:
                prev_body = df['close'].shift(1) - df['open'].shift(1)
                curr_body = df['close'] - df['open']

                return ((prev_body > 0) & 
                        (curr_body < 0) & 
                        (df['open'] > df['close'].shift(1)) & 
                        (df['close'] < df['open'].shift(1))).astype(int)
            except Exception as e:
                logging.error(f"Bearish Engulfing hesaplama hatası: {str(e)}")
                return pd.Series(0, index=df.index)
    def calculate_morning_star(self, df: pd.DataFrame) -> pd.Series:
           """Sabah Yıldızı (Morning Star) formasyonunu hesapla"""
           try:
               result = pd.Series(0, index=df.index)
               for i in range(2, len(df)):
                   if (df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                       abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                       df['close'].iloc[i] > df['open'].iloc[i] and
                       df['close'].iloc[i] > df['open'].iloc[i - 2]):
                       result.iloc[i] = 1
               return result
           except Exception as e:
               logging.error(f"Morning Star hesaplama hatası: {str(e)}")
               return pd.Series(0, index=df.index)
    def calculate_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Akşam Yıldızı (Evening Star) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                    abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                    df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i] < df['open'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Evening Star hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_three_white_soldiers(self, df: pd.DataFrame) -> pd.Series:
        """Üç Beyaz Asker (Three White Soldiers) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i] > df['open'].iloc[i] and
                    df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                    df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                    df['close'].iloc[i] > df['close'].iloc[i - 1] > df['close'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Three White Soldiers hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_three_black_crows(self, df: pd.DataFrame) -> pd.Series:
        """Üç Siyah Karga (Three Black Crows) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and
                    df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                    df['close'].iloc[i] < df['close'].iloc[i - 1] < df['close'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Three Black Crows hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)



    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temel ve gelişmiş teknik indikatörleri hesapla"""
        try:
            logging.info("Calculating technical indicators...")

            # Gerekli sütunları kontrol et
            required_columns = ['high', 'low', 'close', 'open']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logging.error(f"Missing required columns: {missing}")
                return df

            # Minimum veri uzunluğu kontrolü
            if len(df) < 52:  # Ichimoku için minimum 52 periyot gerekli
                logging.warning("Not enough data for calculations")
                return df

            # ---- TEMEL İNDİKATÖRLER ----
            # RSI hesaplama
            df['RSI'] = ta.rsi(df['close'], length=14)

            # MACD hesaplama
            macd_data = ta.macd(df['close'])
            df['MACD'] = macd_data['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd_data['MACDs_12_26_9']
            df['MACD_HIST'] = macd_data['MACDh_12_26_9']

            # Bollinger Bands hesaplama
            bollinger = ta.bbands(df['close'], length=20, std=2)
            df['BB_UPPER'] = bollinger['BBU_20_2.0']
            df['BB_MIDDLE'] = bollinger['BBM_20_2.0']
            df['BB_LOWER'] = bollinger['BBL_20_2.0']

            # Moving Averages
            df['SMA_20'] = ta.sma(df['close'], length=20)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['EMA_200'] = ta.ema(df['close'], length=200)

            # StochRSI hesaplama
            stochrsi = ta.stochrsi(df['close'], length=14)
            df['StochRSI_K'] = stochrsi['STOCHRSIk_14_14_3_3']
            df['StochRSI_D'] = stochrsi['STOCHRSId_14_14_3_3']
            df['StochRSI'] = df['StochRSI_K']

            # ---- GELİŞMİŞ İNDİKATÖRLER ----
            # ADX (Average Directional Index)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['ADX'] = adx['ADX_14']
            df['DI_plus'] = adx['DMP_14']
            df['DI_minus'] = adx['DMN_14']

            # Ichimoku Cloud - Düzeltilmiş versiyon
          # Ichimoku Cloud - Düzeltilmiş versiyon
            # Ichimoku Cloud - Düzeltilmiş versiyon
            try:
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
                if isinstance(ichimoku, pd.DataFrame):
                    logging.info(f"Ichimoku columns: {ichimoku.columns.tolist()}")  # Add logging to check column names
                    ichimoku_columns = {
                        'ISA_9': 'ICHIMOKU_SPAN_A',
                        'ISB_26': 'ICHIMOKU_SPAN_B',
                        'ITS_9': 'ICHIMOKU_CONVERSION',
                        'IKS_26': 'ICHIMOKU_BASE'
                    }
                    for old_name, new_name in ichimoku_columns.items():
                        if old_name in ichimoku.columns:
                            df[new_name] = ichimoku[old_name]
                        else:
                            logging.warning(f"Missing Ichimoku column: {old_name}")  # Add logging for missing columns
            except Exception as e:
                logging.error(f"Ichimoku calculation error: {str(e)}")

            # ---- MUM FORMASYONLARI ----
            df['DOJI'] = self.calculate_doji(df)
            df['HAMMER'] = self.calculate_hammer(df)
            df['BULLISH_ENGULFING'] = self.calculate_bullish_engulfing(df)
            df['BEARISH_ENGULFING'] = self.calculate_bearish_engulfing(df)
            df['MORNING_STAR'] = self.calculate_morning_star(df)
            df['EVENING_STAR'] = self.calculate_evening_star(df)
            df['THREE_WHITE_SOLDIERS'] = self.calculate_three_white_soldiers(df)
            df['THREE_BLACK_CROWS'] = self.calculate_three_black_crows(df)

            # ---- MOMENTUM İNDİKATÖRLERİ ----
            df['ROC'] = ta.roc(df['close'], length=9)
            df['WILLIAMS_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)

            # ---- VOLATILITE İNDİKATÖRLERİ ----
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # ---- HACİM İNDİKATÖRLERİ ----
            if 'volume' in df.columns:
                df['OBV'] = ta.obv(df['close'], df['volume'])
                df['VWAP'] = self.calculate_vwap(df)

            # NaN değerleri temizle
            df = df.ffill().bfill()

            # Hesaplanan göstergeleri kontrol et
            required_indicators = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'SMA_20', 'EMA_20', 'EMA_50', 'EMA_200', 'StochRSI_K', 'StochRSI_D',
                'ADX', 'ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE'
            ]

            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

            if missing_indicators:
                logging.warning(f"Missing indicators after calculation: {missing_indicators}")
            else:
                logging.info("All required indicators calculated successfully")

            return df

        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {str(e)}", exc_info=True)
            return df

    def calculate_doji(self, df: pd.DataFrame) -> pd.Series:
        """Doji mum formasyonunu hesapla"""
        try:
            body = abs(df['close'] - df['open'])
            wick = df['high'] - df['low']
            return (body <= (wick * 0.1)).astype(int)
        except Exception as e:
            logging.error(f"Doji hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Çekiç formasyonunu hesapla"""
        try:
            body = abs(df['close'] - df['open'])
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

            return ((lower_wick > (body * 2)) & (upper_wick <= (body * 0.1))).astype(int)
        except Exception as e:
            logging.error(f"Hammer hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Yutan boğa formasyonunu hesapla"""
        try:
            prev_body = df['close'].shift(1) - df['open'].shift(1)
            curr_body = df['close'] - df['open']

            return ((prev_body < 0) & 
                    (curr_body > 0) & 
                    (df['open'] < df['close'].shift(1)) & 
                    (df['close'] > df['open'].shift(1))).astype(int)
        except Exception as e:
            logging.error(f"Bullish Engulfing hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP (Volume Weighted Average Price) hesapla"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        except Exception as e:
            logging.error(f"VWAP hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)



    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İleri seviye indikatörleri hesapla"""
        try:
            # DataFrame kontrolü
            if df.empty:
                logging.error("DataFrame is empty. Cannot calculate advanced indicators.")
                return df

            # Ichimoku hesaplaması
            try:
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            
                # Ichimoku bileşenlerini ayrı ayrı ekle
                if isinstance(ichimoku, pd.DataFrame):
                    column_mapping = {
                    'ITS_9': 'ICHIMOKU_CONVERSION',
                    'IKS_26': 'ICHIMOKU_BASE',
                    'ISA_26': 'ICHIMOKU_SPAN_A',
                    'ISB_52': 'ICHIMOKU_SPAN_B',
                    'ICS_26': 'ICHIMOKU_CHIKOU'
                    }
                
                    for old_col, new_col in column_mapping.items():
                        if old_col in ichimoku.columns:
                            df[new_col] = ichimoku[old_col]
                        
                logging.info("Ichimoku indicators calculated successfully")
            
            except Exception as ichimoku_error:
                logging.error(f"Ichimoku calculation error: {ichimoku_error}")

            # ADX hesaplaması
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                if isinstance(adx, pd.DataFrame):
                    if 'ADX_14' in adx.columns:
                        df['ADX'] = adx['ADX_14']
                    elif 'ADX' in adx.columns:
                     df['ADX'] = adx['ADX']
                logging.info("ADX calculated successfully")
            
            except Exception as adx_error:
                logging.error(f"ADX calculation error: {adx_error}")

            # NaN değerleri temizle - Update this part
            df = df.ffill().bfill()  # Using the recommended methods instead of fillna
        
            return df

        except Exception as e:
            logging.error(f"İleri seviye indikatör hesaplama hatası: {str(e)}")
            return df
    def _calculate_atr(self, symbol: str) -> float:
        """ATR hesapla"""
        try:
            df = self.get_klines(symbol)
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            return atr.iloc[-1]
        except Exception as e:
            logging.error(f"ATR hesaplama hatası: {e}")
            return 0.0

    def _calculate_dynamic_stop_loss(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik stop loss hesapla"""
        if trade_type == 'BUY':
            return price - (atr * multiplier)
        elif trade_type == 'SELL':
            return price + (atr * multiplier)

    def _calculate_dynamic_take_profit(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik take profit hesapla"""
        if trade_type == 'BUY':
            return price + (atr * multiplier)
        elif trade_type == 'SELL':
            return price - (atr * multiplier)

    async def _place_orders(self, symbol: str, trade_type: str, position_size: float, stop_loss: float, take_profit: float):
        """Order'ları yerleştir"""
        try:
            if trade_type == 'BUY':
                order = self.client.new_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size
                )
            elif trade_type == 'SELL':
                order = self.client.new_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position_size
                )
            # Add stop loss and take profit orders
            sl_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='STOP_MARKET',
                stopPrice=stop_loss,
                quantity=position_size
            )
            tp_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                quantity=position_size
            )
            return order
        except Exception as e:
            logging.error(f"Order yerleştirme hatası: {e}")
            return None

    
    def rsi_strategy(self, df: pd.DataFrame) -> str:
        """RSI Stratejisi"""
        if df['RSI'].iloc[-1] < 30:
            return "BUY"
        elif df['RSI'].iloc[-1] > 70:
            return "SELL"
        return "HOLD"

    def ema_strategy(self, df: pd.DataFrame) -> str:
        """EMA Kesişim Stratejisi"""
        if df['EMA_20'].iloc[-1] > df['SMA_20'].iloc[-1]:
            return "BUY"
        elif df['EMA_20'].iloc[-1] < df['SMA_20'].iloc[-1]:
            return "SELL"
        return "HOLD"

    def bollinger_strategy(self, df: pd.DataFrame) -> str:
        """Bollinger Bands Stratejisi"""
        if df['close'].iloc[-1] < df['BB_LOWER'].iloc[-1]:
            return "BUY"
        elif df['close'].iloc[-1] > df['BB_UPPER'].iloc[-1]:
            return "SELL"
        return "HOLD"

    def hammer_pattern(self, df: pd.DataFrame) -> str:
        """Çekiç (Hammer) formasyonu"""
        for i in range(1, len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            lower_shadow = df['low'].iloc[i] - min(df['open'].iloc[i], df['close'].iloc[i])
            upper_shadow = max(df['open'].iloc[i], df['close'].iloc[i]) - df['high'].iloc[i]
            if lower_shadow > 2 * body and upper_shadow < body and df['close'].iloc[i] > df['open'].iloc[i]:
                return "BUY"
        return "HOLD"

    def dark_cloud_cover(self, df: pd.DataFrame) -> str:
        """Kara Bulut Örtüsü (Dark Cloud Cover) formasyonu"""
        for i in range(1, len(df)):
            if (df['open'].iloc[i] > df['close'].iloc[i - 1] and
                df['close'].iloc[i] < (df['open'].iloc[i - 1] + df['close'].iloc[i - 1]) / 2 and
                df['close'].iloc[i] < df['open'].iloc[i]):
                return "SELL"
        return "HOLD"

    def inverted_hammer(self, df: pd.DataFrame) -> str:
        """Ters Çekiç (Inverted Hammer) formasyonu"""
        for i in range(1, len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            if upper_shadow > 2 * body and lower_shadow < body and df['close'].iloc[i] > df['open'].iloc[i]:
                return "BUY"
        return "HOLD"

    def bullish_engulfing(self, df: pd.DataFrame) -> str:
        """Yutan Boğa (Bullish Engulfing) formasyonu"""
        for i in range(1, len(df)):
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and
                df['close'].iloc[i] > df['open'].iloc[i - 1] and
                df['open'].iloc[i] < df['close'].iloc[i - 1]):
                return "BUY"
        return "HOLD"

    def bearish_engulfing(self, df: pd.DataFrame) -> str:
        """Yutan Ayı (Bearish Engulfing) formasyonu"""
        for i in range(1, len(df)):
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                df['close'].iloc[i] < df['open'].iloc[i - 1] and
                df['open'].iloc[i] > df['close'].iloc[i - 1]):
                return "SELL"
        return "HOLD"

    def doji_pattern(self, df: pd.DataFrame) -> str:
        """Doji formasyonu"""
        for i in range(len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            if body < (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1:
                return "CAUTION"
        return "HOLD"

    def morning_star(self, df: pd.DataFrame) -> str:
        """Sabah Yıldızı (Morning Star) formasyonu"""
        for i in range(2, len(df)):
            if (df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i - 2]):
                return "BUY"
        return "HOLD"

    def three_white_soldiers(self, df: pd.DataFrame) -> str:
        """Üç Beyaz Asker (Three White Soldiers) formasyonu"""
        for i in range(2, len(df)):
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                df['close'].iloc[i] > df['close'].iloc[i - 1] > df['close'].iloc[i - 2]):
                return "BUY"
        return "HOLD"
    

    def generate_ml_signals(self, df: pd.DataFrame) -> dict:
        """ML sinyalleri üret"""
        try:
        # Özellik isimlerini belirterek DataFrame oluştur
            feature_names = ['open', 'high', 'low', 'close', 'volume']
            features = df[feature_names].iloc[-1].to_frame().T
        
            # Ölçeklendirme işlemi
            scaled_features = self.scaler.transform(features)
        
            # Tahmin
            prediction = self.model.predict(scaled_features)
            probability = self.model.predict_proba(scaled_features)[0][prediction[0]]
        
            return {
            'type': 'BUY' if prediction[0] == 1 else 'SELL',
            'probability': probability
        }
        except Exception as e:
            logging.error(f"ML sinyal üretim hatası: {e}")
        return {'type': 'NONE', 'probability': 0.0}

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Teknik analiz sinyalleri üret"""
        try:
            required_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'StochRSI_K', 'StochRSI_D', 'StochRSI'
            ]

            # Gerekli sütunların kontrolü
            missing_columns = [col for col in required_columns if col not in df.columns]
            if df.empty or missing_columns:
                logging.warning(f"Missing columns for signal generation: {missing_columns}")
                return {'type': 'NONE', 'reason': 'missing_data'}

            last_row = df.iloc[-1]
            signals = []
            total_signals = 0

            # Sinyal ağırlıkları
            signal_weights = {
                'RSI': 3,
                'MACD': 2,
                'BB': 2,
                'STOCH': 1,
                'ICHIMOKU': 2,
                'HAMMER': 2,
                'DOJI': 1
            }

            # RSI Sinyali
            if 'RSI' in df.columns:
                total_signals += signal_weights['RSI']
                if last_row['RSI'] < 30:
                    signals.extend(['BUY'] * signal_weights['RSI'])
                elif last_row['RSI'] > 70:
                    signals.extend(['SELL'] * signal_weights['RSI'])

            # MACD Sinyali
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL']):
                total_signals += signal_weights['MACD']
                if last_row['MACD'] > last_row['MACD_SIGNAL']:
                    signals.extend(['BUY'] * signal_weights['MACD'])
                else:
                    signals.extend(['SELL'] * signal_weights['MACD'])

            # Bollinger Bands Sinyali
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER']):
                total_signals += signal_weights['BB']
                if last_row['close'] < last_row['BB_LOWER']:
                    signals.extend(['BUY'] * signal_weights['BB'])
                elif last_row['close'] > last_row['BB_UPPER']:
                    signals.extend(['SELL'] * signal_weights['BB'])

            # StochRSI Sinyali
            if all(col in df.columns for col in ['StochRSI_K', 'StochRSI_D']):
                total_signals += signal_weights['STOCH']
                if last_row['StochRSI_K'] < 20 and last_row['StochRSI_D'] < 20:
                    signals.extend(['BUY'] * signal_weights['STOCH'])
                elif last_row['StochRSI_K'] > 80 and last_row['StochRSI_D'] > 80:
                    signals.extend(['SELL'] * signal_weights['STOCH'])

            # Formasyon Sinyalleri
            # Hammer
            hammer_signal = self.hammer_pattern(df)
            if hammer_signal != "HOLD":
                signals.extend([hammer_signal] * signal_weights['HAMMER'])
                total_signals += signal_weights['HAMMER']

            # Doji
            doji_signal = self.doji_pattern(df)
            if doji_signal != "HOLD":
                signals.extend([doji_signal] * signal_weights['DOJI'])
                total_signals += signal_weights['DOJI']

            # Sinyal analizi ve sonuç üretme
            if signals:
                buy_signals = signals.count('BUY')
                sell_signals = signals.count('SELL')
                total_count = len(signals)

                signal_info = {
                    'buy_count': buy_signals,
                    'sell_count': sell_signals,
                    'total_signals': total_count,
                    'total_indicators': total_signals,
                    'pattern_signals': {
                        'hammer': hammer_signal,
                        'doji': doji_signal
                    }
                }

                # Sinyal kararı
                if buy_signals > sell_signals:
                    return {
                        'type': 'BUY',
                        'strength': buy_signals / total_count,
                        **signal_info
                    }
                elif sell_signals > buy_signals:
                    return {
                        'type': 'SELL',
                        'strength': sell_signals / total_count,
                        **signal_info
                    }

                return {'type': 'HOLD', 'strength': 0, **signal_info}

            return {
                'type': 'HOLD',
                'strength': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_signals': 0,
                'total_indicators': total_signals,
                'pattern_signals': {}
            }

        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}", exc_info=True)
            return {'type': 'NONE', 'reason': 'error'}
        
    def _validate_signals(self, ml_signal: dict, technical_signal: dict) -> bool:
        """Sinyalleri doğrula"""
        try:
            logging.info(f"ML Sinyal: {ml_signal}")
            logging.info(f"Teknik Sinyal: {technical_signal}")
        
            signal_details = (
                f"Sinyal İstatistikleri:\n"
                f"Alış Sinyalleri: {technical_signal.get('buy_count', 0)}\n"
                f"Satış Sinyalleri: {technical_signal.get('sell_count', 0)}\n"
                f"Toplam Sinyal: {technical_signal.get('total_signals', 0)}\n"
                f"Sinyal Gücü: {technical_signal.get('strength', 0):.2f}\n"
                f"Güven Seviyesi: {technical_signal.get('confidence', 0):.2f}\n"
                f"ML Olasılığı: {ml_signal.get('probability', 0):.2f}"
            )
            logging.info(signal_details)
    
            if technical_signal['type'] in ['BUY', 'SELL']:
                signal_strength = technical_signal.get('strength', 0)
                signal_confidence = technical_signal.get('confidence', 0)
        
                if (ml_signal['type'] == technical_signal['type'] and 
                    signal_strength > 0.6 and
                    signal_confidence > 0.5 and
                    ml_signal['probability'] > 0.55):
            
                    logging.info(f"✅ Sinyal Onaylandı: {technical_signal['type']}\n{signal_details}")
                    return True
            
            return False
    
        except Exception as e:
            logging.error(f"Sinyal doğrulama hatası: {e}")
            return False

    def is_trading_allowed(self) -> bool:
        """Trading koşullarını kontrol et"""
        current_hour = datetime.now().hour
        if not (self.config['trading_hours']['start_hour'] <= 
                current_hour < self.config['trading_hours']['end_hour']):
            return False
            
        if self.daily_trades >= self.config['risk_management']['max_trades_per_day']:
            return False
            
        return True

    def calculate_position_size(self, symbol: str, current_price: float) -> float:
        """Pozisyon büyüklüğünü hesapla"""
        try:
            # Bakiyeyi al
            balance = float(self.get_account_balance())
            logging.info(f"Mevcut bakiye: {balance} USDT")
        
            # Minimum işlem miktarı (örnek: 0.001 BTC için yaklaşık 0.05 USDT)
            min_trade_value = 0.05
        
            # Risk miktarını hesapla (bakiyenin %95'i)
            risk_amount = balance * 0.95
        
            # Pozisyon büyüklüğünü hesapla
            position_size = risk_amount / current_price
        
            # Minimum işlem değeri kontrolü
            if position_size * current_price < min_trade_value:
                logging.warning(f"İşlem değeri çok düşük: {position_size * current_price} USDT")
                return 0
            
            return position_size
        
        except Exception as e:
            logging.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0
        
    def get_symbol_info(self, symbol: str) -> dict:
        """Sembol bilgilerini al"""
        try:
            exchange_info = self.client.exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return {
                        'pricePrecision': s['pricePrecision'],
                        'quantityPrecision': s['quantityPrecision'],
                        'minQty': float(next(f['minQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE')),
                        'maxQty': float(next(f['maxQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE')),
                        'stepSize': float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                    }
            return None
        except Exception as e:
            logging.error(f"Sembol bilgisi alma hatası: {e}")
            return None

    def round_to_precision(self, value: float, precision: int) -> float:
        """Değeri belirtilen hassasiyete yuvarla"""
        factor = 10 ** precision
        return float(round(value * factor) / factor)

    async def execute_trade_with_risk_management(self, symbol: str, signal_type: str, current_price: float):
        """İşlem yönetimi ve risk kontrolü"""
        try:
            trade_side = signal_type

            # Hesap bakiyesini al
            balance = float(self.get_account_balance())
            logging.info(f"Mevcut bakiye: {balance} USDT")

            # Check if balance is below 5 USD
            if balance < 5.0:
                logging.warning(f"Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                await self.send_telegram(f"⚠️ Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                return False

            # Kaldıraç ayarı
            try:
                self.client.change_leverage(
                    symbol=symbol,
                    leverage=5
                )
                logging.info(f"Kaldıraç ayarlandı: {symbol} 5x")
            except Exception as e:
                logging.error(f"Kaldıraç ayarlama hatası: {e}")
                return False

            # Sembol bilgilerini al
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logging.error(f"Sembol bilgisi alınamadı: {symbol}")
                return False

            # Minimum işlem değeri (5.1 USDT) için quantity hesaplama
            min_notional = 5.2  # Biraz daha yüksek tutalım
            min_quantity = min_notional / current_price

            # Risk bazlı quantity hesaplama
            risk_percentage = 0.95
            risk_based_quantity = (balance * risk_percentage) / current_price

            # İkisinden büyük olanı seç
            quantity = max(min_quantity, risk_based_quantity)

            # Quantity'yi sembol hassasiyetine yuvarla
            quantity = self.round_to_precision(quantity, symbol_info['quantityPrecision'])
            price = self.round_to_precision(current_price, symbol_info['pricePrecision'])

            # Son kontrol
            final_notional = quantity * price
            logging.info(f"Final işlem değeri: {final_notional} USDT")

            if final_notional < min_notional:
            # Quantity'yi tekrar ayarla
                quantity = self.round_to_precision((min_notional / price) * 1.01, symbol_info['quantityPrecision'])
                final_notional = quantity * price
                logging.info(f"Quantity yeniden ayarlandı: {quantity} ({final_notional} USDT)")

            # Market emri oluştur
            try:
                order = self.client.new_order(
                    symbol=symbol,
                    side=trade_side,
                    type='MARKET',
                    quantity=quantity
                )

                # Stop Loss ve Take Profit hesapla
                sl_price = price * (0.98 if trade_side == 'BUY' else 1.02)
                tp_price = price * (1.03 if trade_side == 'BUY' else 0.97)

                # Stop Loss emri
                sl_order = self.client.new_order(
                    symbol=symbol,
                    side='SELL' if trade_side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=self.round_to_precision(sl_price, symbol_info['pricePrecision']),
                    closePosition='true'
                )

                # Take Profit emri
                tp_order = self.client.new_order(
                    symbol=symbol,
                    side='SELL' if trade_side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=self.round_to_precision(tp_price, symbol_info['pricePrecision']),
                    closePosition='true'
                )

                message = (
                    f"✅ İşlem Gerçekleşti\n"
                    f"Sembol: {symbol}\n"
                    f"Yön: {trade_side}\n"
                    f"Miktar: {quantity}\n"
                    f"Fiyat: {price}\n"
                    f"İşlem Değeri: {final_notional:.2f} USDT\n"
                    f"Stop Loss: {sl_price}\n"
                    f"Take Profit: {tp_price}\n"
                    f"Kaldıraç: 5x\n"
                    f"Bakiye: {balance} USDT"
                )

                logging.info(f"İşlem başarılı: {symbol} {trade_side} {quantity}")
                await self.send_telegram(message)

                return True

            except Exception as order_error:
                logging.error(f"Order yerleştirme hatası: {order_error}")
                await self.send_telegram(f"⚠️ İşlem Hatası: {symbol} - {str(order_error)}")
                return False

        except Exception as e:
            logging.error(f"İşlem yönetimi hatası: {e}")
            await self.send_telegram(f"⚠️ İşlem Yönetimi Hatası: {symbol} - {str(e)}")
            return False
    
    def get_account_balance(self):
        """Hesap bakiyesini al (Vadeli işlemler hesabı)"""
        try:
            account = self.client.futures_account_balance()
            for asset in account:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            logging.error(f"Bakiye alma hatası: {e}")
            return 0.0
          
    async def _send_trade_notification(self, symbol, signal, price, size, sl, tp):
        """Trade bildirimini gönder"""
        message = (
            f"🤖 Trade Executed\n"
            f"Symbol: {symbol}\n"
            f"Type: {signal['type']}\n"
            f"Price: {price:.8f}\n"
            f"Size: {size:.8f}\n"
            f"Stop Loss: {sl:.8f}\n"
            f"Take Profit: {tp:.8f}\n"
            f"Probability: {signal['probability']:.2f}"
        )
        await self.send_telegram(message)


    async def run(self):
        """Ana bot döngüsü"""
        try:
            logging.info(f"Bot started by {self.config.get('created_by', 'unknown')}")
            await self.send_telegram("🚀 Trading Bot Activated")
    
            while True:
                try:
                    # Trading saatleri kontrolü
                    if self.is_trading_allowed():
                        for symbol in self.config['symbols']:
                            # Mum verilerini al
                            df = self.get_klines(symbol)
                            if df.empty:
                                logging.warning(f"No data received for {symbol}")
                                continue

                                # Temel göstergeleri hesapla
                            df = self.calculate_indicators(df)
                            logging.info(f"Basic indicators calculated for {symbol}")

                            # İleri seviye göstergeleri hesapla
                            df = self.calculate_advanced_indicators(df)
                            logging.info(f"Advanced indicators calculated for {symbol}")

                            # ML ve teknik sinyalleri üret
                            ml_signal = self.generate_ml_signals(df)
                            technical_signal = self.generate_signals(df)

                            # Sinyalleri doğrula
                            if self._validate_signals(ml_signal, technical_signal):
                                current_price = float(df['close'].iloc[-1])
                                logging.info(f"Sinyal onaylandı: {ml_signal['type']} (Güç: {technical_signal['strength']}, ML Olasılık: {ml_signal['probability']})")
                            
                                # Burada signal_type olarak sadece string gönderiyoruz
                                await self.execute_trade_with_risk_management(
                                    symbol=symbol,
                                    signal_type=ml_signal['type'],  # Sadece 'BUY' veya 'SELL' string'i
                                    current_price=current_price
                                )

                            # Rate limit kontrolü
                            await asyncio.sleep(self.rate_limit_delay)

                    # Günlük istatistikleri sıfırla
                    if datetime.now().date() > self.last_daily_reset:
                        self.reset_daily_stats()

                    # Ana döngü bekleme süresi
                    await asyncio.sleep(self.config['check_interval'])

                except Exception as loop_error:
                    logging.error(f"Loop iteration error: {loop_error}")
                    await self.send_telegram(f"⚠️ Error in main loop: {loop_error}")
                    await asyncio.sleep(60)

        except Exception as e:
            logging.error(f"Critical error in run method: {e}")
            await self.send_telegram("🚨 Bot stopped due to critical error!")
            raise

if __name__ == "__main__":
    # Logging ayarları
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='advanced_trading_bot.log'
    )

    try:
        # Bot instance'ını oluştur
        bot = BinanceFuturesBot()
        
        # Modern asyncio kullanımı
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {e}")
