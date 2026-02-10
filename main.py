from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ GLOBAL STATE ============
bot_enabled = False
demo_account = {
    "balance": 100000.0,
    "positions": []
}
trade_journal = []
strategies = {}
active_strategy_id = None

# ============ DATA MODELS ============
class Strategy(BaseModel):
    name: str
    market: str
    timeframe: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    stop_loss_pct: float
    take_profit_pct: float

class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: float
    price: float

class BotToggle(BaseModel):
    enabled: bool

# ============ MARKET DATA (REAL, 15-MIN DELAYED) ============
def get_real_market_data(symbol: str, interval: str = "15m", period: str = "1d"):
    """Fetch real market data using yfinance (15-min delayed)"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            return None
        
        data = {
            "symbol": symbol,
            "current_price": float(df['Close'].iloc[-1]),
            "open": float(df['Open'].iloc[-1]),
            "high": float(df['High'].iloc[-1]),
            "low": float(df['Low'].iloc[-1]),
            "volume": int(df['Volume'].iloc[-1]),
            "timestamp": df.index[-1].isoformat(),
            "history": df.to_dict('records')
        }
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ============ TECHNICAL INDICATORS ============
def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: List[float]) -> Dict:
    """Calculate MACD"""
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0}
    
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    
    # Simple approximation for signal line
    signal_line = macd_line * 0.8
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }

def analyze_symbol(symbol: str) -> Optional[Dict]:
    """Analyze a symbol and return signals"""
    data = get_real_market_data(symbol)
    if not data or not data.get('history'):
        return None
    
    history = data['history']
    if len(history) < 20:
        return None
    
    closes = [candle['Close'] for candle in history]
    volumes = [candle['Volume'] for candle in history]
    
    # Calculate indicators
    rsi = calculate_rsi(closes)
    ema_9 = calculate_ema(closes, 9)
    ema_21 = calculate_ema(closes, 21)
    macd = calculate_macd(closes)
    avg_volume = sum(volumes[-20:]) / 20
    current_volume = volumes[-1]
    
    # Generate signal
    signal = "HOLD"
    reasons = []
    
    if rsi < 30 and ema_9 > ema_21 and current_volume > avg_volume * 1.2:
        signal = "BUY"
        reasons.append("RSI oversold")
        reasons.append("EMA bullish crossover")
        reasons.append("High volume")
    elif rsi > 70 and ema_9 < ema_21:
        signal = "SELL"
        reasons.append("RSI overbought")
        reasons.append("EMA bearish crossover")
    elif rsi < 35:
        signal = "BUY"
        reasons.append("RSI oversold")
    elif rsi > 65 and macd['histogram'] < 0:
        signal = "SELL"
        reasons.append("RSI high + MACD bearish")
    
    return {
        "symbol": symbol,
        "price": data['current_price'],
        "signal": signal,
        "confidence": "HIGH" if len(reasons) > 1 else "MEDIUM",
        "indicators": {
            "rsi": round(rsi, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "macd": round(macd['macd'], 2),
            "volume_ratio": round(current_volume / avg_volume, 2)
        },
        "reasons": reasons,
        "timestamp": datetime.now().isoformat()
    }

# ============ TRADING LOGIC ============
def execute_trade(symbol: str, action: str, quantity: float, price: float):
    """Execute a trade in the demo account"""
    global demo_account, trade_journal
    
    trade = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "timestamp": datetime.now().isoformat(),
        "strategy": active_strategy_id
    }
    
    if action == "BUY":
        cost = quantity * price
        if demo_account["balance"] >= cost:
            demo_account["balance"] -= cost
            demo_account["positions"].append({
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": price,
                "timestamp": trade["timestamp"]
            })
            trade_journal.append(trade)
            return {"success": True, "trade": trade}
    
    elif action == "SELL":
        for position in demo_account["positions"]:
            if position["symbol"] == symbol:
                revenue = quantity * price
                demo_account["balance"] += revenue
                
                # Calculate P&L
                pnl = (price - position["entry_price"]) * quantity
                trade["pnl"] = pnl
                
                demo_account["positions"].remove(position)
                trade_journal.append(trade)
                return {"success": True, "trade": trade}
    
    return {"success": False, "message": "Trade failed"}

# ============ BOT LOOP ============
async def bot_trading_loop():
    """Automated trading bot that runs when enabled"""
    global bot_enabled
    
    # Default symbols to monitor
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC-USD", "ETH-USD"]
    
    while True:
        if bot_enabled:
            print(f"[BOT] Running scan at {datetime.now()}")
            
            for symbol in symbols:
                try:
                    analysis = analyze_symbol(symbol)
                    if analysis and analysis["signal"] != "HOLD":
                        # Execute trade based on signal
                        price = analysis["price"]
                        quantity = 1  # Simple quantity for demo
                        
                        if analysis["signal"] == "BUY":
                            result = execute_trade(symbol, "BUY", quantity, price)
                            if result["success"]:
                                print(f"[BOT] Bought {quantity} {symbol} at ${price}")
                        
                        elif analysis["signal"] == "SELL":
                            result = execute_trade(symbol, "SELL", quantity, price)
                            if result["success"]:
                                print(f"[BOT] Sold {quantity} {symbol} at ${price}")
                
                except Exception as e:
                    print(f"[BOT] Error trading {symbol}: {e}")
                
                await asyncio.sleep(1)  # Rate limiting
        
        await asyncio.sleep(30)  # Check every 30 seconds

# ============ API ENDPOINTS ============

@app.get("/")
def root():
    return {"status": "Trading Scanner API Online", "bot_enabled": bot_enabled}

@app.get("/account")
def get_account():
    """Get demo account status"""
    return {
        "balance": demo_account["balance"],
        "positions": demo_account["positions"],
        "total_value": demo_account["balance"] + sum(
            p["quantity"] * p["entry_price"] for p in demo_account["positions"]
        )
    }

@app.get("/scan/{symbol}")
def scan_symbol(symbol: str):
    """Scan a single symbol"""
    analysis = analyze_symbol(symbol)
    if analysis:
        return analysis
    return {"error": f"Could not analyze {symbol}"}

@app.get("/scan")
def scan_multiple():
    """Scan multiple symbols"""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "BTC-USD", "ETH-USD"]
    results = []
    
    for symbol in symbols:
        analysis = analyze_symbol(symbol)
        if analysis:
            results.append(analysis)
    
    return {"scanned_at": datetime.now().isoformat(), "results": results}

@app.post("/trade")
def trade(trade_req: TradeRequest):
    """Execute a manual trade"""
    result = execute_trade(
        trade_req.symbol,
        trade_req.action,
        trade_req.quantity,
        trade_req.price
    )
    return result

@app.get("/journal")
def get_journal():
    """Get trade journal"""
    return {
        "trades": trade_journal,
        "total_trades": len(trade_journal),
        "profitable_trades": sum(1 for t in trade_journal if t.get("pnl", 0) > 0)
    }

@app.post("/bot/toggle")
def toggle_bot(toggle: BotToggle):
    """Toggle bot on/off"""
    global bot_enabled
    bot_enabled = toggle.enabled
    return {"bot_enabled": bot_enabled, "message": f"Bot {'enabled' if bot_enabled else 'disabled'}"}

@app.get("/bot/status")
def bot_status():
    """Get bot status"""
    return {"bot_enabled": bot_enabled}

@app.post("/strategy")
def create_strategy(strategy: Strategy):
    """Create a new trading strategy"""
    strategy_id = f"strategy_{len(strategies) + 1}"
    strategies[strategy_id] = strategy.dict()
    return {"strategy_id": strategy_id, "strategy": strategies[strategy_id]}

@app.get("/strategies")
def list_strategies():
    """List all strategies"""
    return {"strategies": strategies}

@app.post("/strategy/{strategy_id}/activate")
def activate_strategy(strategy_id: str):
    """Activate a strategy"""
    global active_strategy_id
    if strategy_id in strategies:
        active_strategy_id = strategy_id
        return {"active_strategy": strategy_id}
    return {"error": "Strategy not found"}

# ============ WEBSOCKET FOR REAL-TIME UPDATES ============
@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket for real-time signal updates"""
    await websocket.accept()
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC-USD", "ETH-USD"]
    
    try:
        while True:
            signals = []
            for symbol in symbols:
                analysis = analyze_symbol(symbol)
                if analysis:
                    signals.append(analysis)
            
            await websocket.send_json({
                "timestamp": datetime.now().isoformat(),
                "bot_enabled": bot_enabled,
                "signals": signals
            })
            
            await asyncio.sleep(60)  # Update every 60 seconds (respect rate limits)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# ============ STARTUP ============
@app.on_event("startup")
async def startup_event():
    """Start the bot loop on startup"""
    asyncio.create_task(bot_trading_loop())
    print("Trading Scanner API started")
    print("Bot trading loop started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
