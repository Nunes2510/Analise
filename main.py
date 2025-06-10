import os
import time
import requests
import pandas as pd
import numpy as np
from binance.client import Client

# ===================== CONFIG =====================
API_KEY = os.getenv("BINANCE_API")
API_SECRET = os.getenv("BINANCE_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace token (opcional)
HF_MODEL = "HuggingFaceH4/zephyr-7b-alpha"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

client = Client(API_KEY, API_SECRET)

# ===================== INDICADORES =====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[low.diff() > 0] = 0

    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

# ===================== ANÁLISE TÉCNICA =====================
def fetch_klines(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval='1d', limit=90)
        df = pd.DataFrame(klines, columns=['time','open','high','low','close','volume','ct','qav','ntrades','tbav','tqav','ignore'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        return df
    except:
        return None

def score_coin(symbol):
    df = fetch_klines(symbol)
    if df is None or len(df) < 60:
        return None

    rsi = compute_rsi(df['close'])
    adx = compute_adx(df)
    upper, lower = compute_bollinger_bands(df['close'])
    bb_width = (upper - lower) / df['close']

    latest_rsi = rsi.iloc[-1]
    latest_adx = adx.iloc[-1]
    latest_bb = bb_width.iloc[-1]

    volume = df['volume']
    vol_growth = volume[-3:].mean() > volume[-30:].mean()
    breakout = df['close'].iloc[-1] > df['high'].rolling(5).max().iloc[-2]
    sideways = (df['close'][-60:].max() - df['close'][-60:].min()) / df['close'][-60:].min() < 0.10

    score = 0
    if 40 <= latest_rsi <= 50: score += 2
    if latest_bb < 0.10: score += 2
    if latest_adx > 20: score += 2
    if vol_growth: score += 2
    if breakout: score += 2
    if sideways: score += 1

    return {"symbol": symbol, "score": score, "rsi": latest_rsi, "adx": latest_adx, "bb_width": latest_bb}

# ===================== HUGGINGFACE INTEGRAÇÃO =====================
def perguntar_para_ia(dados):
    prompt = f"""
    A moeda {dados['symbol']} teve RSI {dados['rsi']:.2f}, ADX {dados['adx']:.2f}, volume crescente e bandas de Bollinger estreitas.
    Rompeu a máxima dos últimos dias. Parece que ela tem chance de subir 20% rapidamente?
    Explique de forma simples.
    """
    response = requests.post(HF_URL, headers=HEADERS, json={"inputs": prompt})
    try:
        return response.json()[0]['generated_text']
    except:
        return "Erro na IA"

# ===================== EXECUÇÃO PRINCIPAL =====================
def main():
    tickers = client.get_all_tickers()
    usdt_pairs = [t['symbol'] for t in tickers if t['symbol'].endswith('USDT') and 'DOWN' not in t['symbol']]

    analisadas = []
    for symbol in usdt_pairs:
        try:
            result = score_coin(symbol)
            if result and result['score'] >= 6:
                print(f"\nAnalyzing {symbol}...")
                resposta = perguntar_para_ia(result)
                result['ia_resposta'] = resposta
                analisadas.append(result)
        except Exception as e:
            print(f"Erro em {symbol}: {e}")

    if not analisadas:
        print("Nenhuma moeda promissora encontrada.")
        return

    melhores = sorted(analisadas, key=lambda x: -x['score'])
    escolha = melhores[0]
    print("\n=== MELHOR MOEDA ESCOLHIDA ===")
    print(f"Moeda: {escolha['symbol']}")
    print(f"Score: {escolha['score']}")
    print(f"RSI: {escolha['rsi']:.2f}")
    print(f"ADX: {escolha['adx']:.2f}")
    print(f"Bollinger Width: {escolha['bb_width']:.4f}")
    print(f"Resposta da IA: {escolha['ia_resposta']}")

    # Salvar histórico
    pd.DataFrame(melhores).to_csv("analises_resultado.csv", index=False)

if __name__ == '__main__':
    main()
