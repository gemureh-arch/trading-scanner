from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np

# Finnhub API configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ GLOBAL STATE ============
demo_account = {
    "balance": 100000.0,
    "positions": [],
    "trade_history": []
}

strategies = {
    "default": {
        "name": "RSI Momentum",
        "description": "Buy when RSI < 30, Sell when RSI > 70",
        "rules": {
            "entry": [{"indicator": "RSI", "operator": "<", "value": 30}],
            "exit": [{"indicator": "RSI", "operator": ">", "value": 70}]
        }
    }
}

active_strategy = "default"
bot_enabled = False

# Track connected WebSocket clients
connected_clients: List[WebSocket] = []

# ============ MODELS ============
class TradeRequest(BaseModel):
    symbol: str
    action: str
    price: float
    quantity: float = 1.0

class StrategyCreate(BaseModel):
    name: str
    description: str
    rules: dict

class BotToggle(BaseModel):
    enabled: bool

# ============ MARKET DATA ============
# ============ MARKET DATA (FINNHUB) ============
def get_real_market_data(symbol: str, interval: str = "D", period_days: int = 30):
    """Fetch real market data using Finnhub API"""
    try:
        if not FINNHUB_API_KEY:
            return None
        
        # Get current quote (real-time price)
        quote_url = f"{FINNHUB_BASE_URL}/quote"
        quote_params = {
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        }
        quote_response = requests.get(quote_url, params=quote_params, timeout=10)
        
        if quote_response.status_code != 200:
            return None
        
        quote_data = quote_response.json()
        current_price = quote_data.get("c", 0)  # Current price
        
        if current_price == 0:
            return None
        
        # Get historical candles for indicators
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=period_days)).timestamp())
        
        candles_url = f"{FINNHUB_BASE_URL}/stock/candle"
        candles_params = {
            "symbol": symbol,
            "resolution": interval,  # D = daily, 60 = hourly, 15 = 15min
            "from": start_time,
            "to": end_time,
            "token": FINNHUB_API_KEY
        }
        candles_response = requests.get(candles_url, params=candles_params, timeout=10)
        
        if candles_response.status_code != 200:
            return None
        
        candles_data = candles_response.json()
        
        if candles_data.get("s") != "ok":
            return None
        
        # Convert to arrays for calculations
        closes = np.array(candles_data.get("c", []))
        volumes = np.array(candles_data.get("v", []))
        
        if len(closes) < 14:
            return None
        
        # Calculate indicators
        rsi = calculate_rsi(closes)
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        
        return {
            "symbol": symbol,
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "indicators": {
                "RSI": rsi,
                "EMA_9": ema_9,
                "EMA_21": ema_21,
                "Volume": float(current_volume),
                "Avg_Volume": float(avg_volume)
            },
            "data": {
                "closes": closes[-50:].tolist(),  # Last 50 candles
                "volumes": volumes[-50:].tolist()
            }
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def calculate_ema(prices, period):
    """Calculate EMA indicator"""
    if len(prices) < period:
        return float(np.mean(prices))
    
    ema = prices[0]
    multiplier = 2 / (period + 1)
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return float(ema)

# ============ STRATEGY ENGINE ============
def evaluate_strategy(symbol: str, market_data: dict, strategy: dict) -> str:
    """Evaluate if strategy conditions are met"""
    if not market_data or "indicators" not in market_data:
        return "HOLD"
    
    indicators = market_data["indicators"]
    rules = strategy.get("rules", {})
    
    # Check entry conditions (BUY signals)
    entry_conditions = rules.get("entry", [])
    if all(check_condition(indicators, cond) for cond in entry_conditions):
        return "BUY"
    
    # Check exit conditions (SELL signals)
    exit_conditions = rules.get("exit", [])
    if all(check_condition(indicators, cond) for cond in exit_conditions):
        return "SELL"
    
    return "HOLD"

def check_condition(indicators: dict, condition: dict) -> bool:
    """Check if a single condition is met"""
    indicator_name = condition.get("indicator")
    operator = condition.get("operator")
    value = condition.get("value")
    
    if indicator_name not in indicators:
        return False
    
    indicator_value = indicators[indicator_name]
    
    if operator == "<":
        return indicator_value < value
    elif operator == ">":
        return indicator_value > value
    elif operator == "<=":
        return indicator_value <= value
    elif operator == ">=":
        return indicator_value >= value
    elif operator == "==":
        return indicator_value == value
    
    return False

# ============ TRADING EXECUTION ============
def execute_trade(symbol: str, action: str, price: float, quantity: float = 1.0):
    """Execute a trade in demo account"""
    global demo_account
    
    trade_value = price * quantity
    
    if action == "BUY":
        if demo_account["balance"] >= trade_value:
            demo_account["positions"].append({
                "symbol": symbol,
                "action": action,
                "price": price,
                "quantity": quantity,
                "timestamp": datetime.now().isoformat()
            })
            demo_account["balance"] -= trade_value
            
            trade_record = {
                "symbol": symbol,
                "action": action,
                "price": price,
                "quantity": quantity,
                "value": trade_value,
                "timestamp": datetime.now().isoformat(),
                "balance_after": demo_account["balance"]
            }
            demo_account["trade_history"].append(trade_record)
            return True
    
    elif action == "SELL":
        # Find matching position to sell
        for i, pos in enumerate(demo_account["positions"]):
            if pos["symbol"] == symbol and pos["action"] == "BUY":
                demo_account["positions"].pop(i)
                demo_account["balance"] += trade_value
                
                profit = trade_value - (pos["price"] * pos["quantity"])
                
                trade_record = {
                    "symbol": symbol,
                    "action": action,
                    "price": price,
                    "quantity": quantity,
                    "value": trade_value,
                    "profit": profit,
                    "timestamp": datetime.now().isoformat(),
                    "balance_after": demo_account["balance"]
                }
                demo_account["trade_history"].append(trade_record)
                return True
    
    return False

# ============ BOT LOGIC ============
async def bot_loop():
    """Main bot loop that runs when bot is enabled"""
    global bot_enabled
    
    # Symbols to scan
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # Crypto needs different format
    
    while True:
        if bot_enabled:
            strategy = strategies.get(active_strategy)
            
            for symbol in symbols:
                try:
                    # Use daily candles for reliable data
                    market_data = get_real_market_data(symbol, interval="D", period_days=30)
                    if market_data:
                        signal = evaluate_strategy(symbol, market_data, strategy)
                        
                        if signal in ["BUY", "SELL"]:
                            price = market_data["price"]
                            success = execute_trade(symbol, signal, price, 1.0)
                            
                            if success:
                                # Broadcast to all connected clients
                                await broadcast_message({
                                    "type": "trade_executed",
                                    "data": {
                                        "symbol": symbol,
                                        "action": signal,
                                        "price": price,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                })
                
                except Exception as e:
                    print(f"Bot error for {symbol}: {str(e)}")
                
                # Small delay between requests to respect rate limits
                await asyncio.sleep(2)
        
        await asyncio.sleep(3600)  # Check every hour

async def broadcast_message(message: dict):
    """Send message to all connected WebSocket clients"""
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

# ============ API ENDPOINTS ============
@app.get("/")
def root():
    return {"status": "Trading Scanner API Running", "version": "1.0.0"}

@app.get("/account")
def get_account():
    """Get current account status"""
    return demo_account

@app.post("/trade")
def manual_trade(trade: TradeRequest):
    """Execute a manual trade"""
    success = execute_trade(trade.symbol, trade.action, trade.price, trade.quantity)
    return {
        "success": success,
        "account": demo_account
    }

@app.get("/market/{symbol}")
def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    data = get_real_market_data(symbol)
    if data:
        strategy = strategies.get(active_strategy)
        signal = evaluate_strategy(symbol, data, strategy)
        data["signal"] = signal
        return data
    return {"error": "Unable to fetch market data"}

@app.get("/scan")
def scan_markets():
    """Scan multiple markets and return signals"""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    strategy = strategies.get(active_strategy)
    results = []
    
    for symbol in symbols:
        market_data = get_real_market_data(symbol, interval="D", period_days=30)
        if market_data:
            signal = evaluate_strategy(symbol, market_data, strategy)
            results.append({
                "symbol": symbol,
                "price": market_data["price"],
                "signal": signal,
                "indicators": market_data["indicators"]
            })
    
    return {"results": results, "strategy": active_strategy}

@app.get("/strategies")
def get_strategies():
    """Get all available strategies"""
    return {
        "strategies": strategies,
        "active": active_strategy
    }

@app.post("/strategies")
def create_strategy(strategy: StrategyCreate):
    """Create a new strategy"""
    strategies[strategy.name] = {
        "name": strategy.name,
        "description": strategy.description,
        "rules": strategy.rules
    }
    return {"success": True, "strategy": strategy.name}

@app.post("/strategies/select/{strategy_name}")
def select_strategy(strategy_name: str):
    """Select active strategy"""
    global active_strategy
    if strategy_name in strategies:
        active_strategy = strategy_name
        return {"success": True, "active_strategy": active_strategy}
    return {"success": False, "error": "Strategy not found"}

@app.get("/bot/status")
def get_bot_status():
    """Get current bot status"""
    return {"enabled": bot_enabled, "strategy": active_strategy}

@app.post("/bot/toggle")
def toggle_bot(toggle: BotToggle):
    """Toggle bot on/off"""
    global bot_enabled
    bot_enabled = toggle.enabled
    return {"success": True, "enabled": bot_enabled}

@app.get("/journal")
def get_journal():
    """Get trading journal (trade history)"""
    return {
        "trades": demo_account["trade_history"],
        "total_trades": len(demo_account["trade_history"])
    }

@app.post("/backtest/{symbol}")
async def backtest_symbol(symbol: str, days: int = 30):
    """Backtest current strategy on a symbol"""
    try:
        # Get historical data
        market_data = get_real_market_data(symbol, interval="D", period_days=days)
        
        if not market_data or "data" not in market_data:
            return {"error": "No data available"}
        
        closes = market_data["data"]["closes"]
        
        if len(closes) < 14:
            return {"error": "Not enough historical data"}
        
        strategy = strategies.get(active_strategy)
        trades = []
        balance = 100000.0
        position = None
        
        for i in range(14, len(closes)):
            # Calculate indicators for this point
            prices_slice = closes[:i+1]
            rsi = calculate_rsi(np.array(prices_slice))
            ema_9 = calculate_ema(np.array(prices_slice), 9)
            ema_21 = calculate_ema(np.array(prices_slice), 21)
            
            indicators = {
                "RSI": rsi,
                "EMA_9": ema_9,
                "EMA_21": ema_21,
                "Volume": 0,
                "Avg_Volume": 0
            }
            
            test_data = {"indicators": indicators, "price": closes[i]}
            signal = evaluate_strategy(symbol, test_data, strategy)
            
            if signal == "BUY" and position is None and balance >= closes[i]:
                position = {"entry_price": closes[i], "entry_index": i}
                balance -= closes[i]
                trades.append({
                    "action": "BUY",
                    "price": float(closes[i]),
                    "index": i
                })
            
            elif signal == "SELL" and position is not None:
                exit_price = closes[i]
                profit = exit_price - position["entry_price"]
                balance += exit_price
                
                trades.append({
                    "action": "SELL",
                    "price": float(exit_price),
                    "profit": float(profit),
                    "index": i
                })
                position = None
        
        # Calculate stats
        winning_trades = [t for t in trades if t.get("profit", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit", 0) < 0]
        total_profit = sum(t.get("profit", 0) for t in trades)
        
        return {
            "symbol": symbol,
            "strategy": active_strategy,
            "initial_balance": 100000.0,
            "final_balance": balance,
            "total_profit": total_profit,
            "total_trades": len(trades) // 2,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1) * 100,
            "trades": trades
        }
    
    except Exception as e:
        return {"error": str(e)}

# ============ WEBSOCKET ============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Send current account status
            await websocket.send_json({
                "type": "account_update",
                "data": demo_account
            })
    
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

bash

cat > /tmp/debug_endpoint.txt << 'EOF'
# Add this endpoint to your main.py (before the startup event)

@app.get("/debug/env")
def debug_env():
    """Check if API key is loaded"""
    return {
        "api_key_set": bool(FINNHUB_API_KEY),
        "api_key_length": len(FINNHUB_API_KEY) if FINNHUB_API_KEY else 0,
        "api_key_preview": FINNHUB_API_KEY[:5] + "..." if FINNHUB_API_KEY else "NOT SET"
    }

@app.get("/debug/market/{symbol}")
def debug_market(symbol: str):
    """Debug version with full error details"""
    import traceback
    try:
        if not FINNHUB_API_KEY:
            return {"error": "API key not set", "key_var": FINNHUB_API_KEY}
        
        # Test quote endpoint
        quote_url = f"{FINNHUB_BASE_URL}/quote"
        quote_params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        
        response = requests.get(quote_url, params=quote_params, timeout=10)
        
        return {
            "status_code": response.status_code,
            "response": response.json(),
            "url": quote_url,
            "symbol": symbol
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
EOF
cat /tmp/debug_endpoint.txt
Output

# Add this endpoint to your main.py (before the startup event)

@app.get("/debug/env")
def debug_env():
    """Check if API key is loaded"""
    return {
        "api_key_set": bool(FINNHUB_API_KEY),
        "api_key_length": len(FINNHUB_API_KEY) if FINNHUB_API_KEY else 0,
        "api_key_preview": FINNHUB_API_KEY[:5] + "..." if FINNHUB_API_KEY else "NOT SET"
    }

@app.get("/debug/market/{symbol}")
def debug_market(symbol: str):
    """Debug version with full error details"""
    import traceback
    try:
        if not FINNHUB_API_KEY:
            return {"error": "API key not set", "key_var": FINNHUB_API_KEY}
        
        # Test quote endpoint
        quote_url = f"{FINNHUB_BASE_URL}/quote"
        quote_params = {"symbol": symbol, "token": FINNHUB_API_KEY}
        
        response = requests.get(quote_url, params=quote_params, timeout=10)
        
        return {
            "status_code": response.status_code,
            "response": response.json(),
            "url": quote_url,
            "symbol": symbol
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
# ============ STARTUP ============
@app.on_event("startup")
async def startup_event():
    """Start bot loop on startup"""
    asyncio.create_task(bot_loop())
    print("🚀 Trading Scanner Backend Started")
    print("📊 Bot loop initialized")
    print(f"💰 Demo account balance: ${demo_account['balance']}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
