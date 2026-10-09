import os
import discord
from discord import app_commands
from discord.ext import commands, tasks
import yfinance as yf
from groq import Groq
import asyncio
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, time
import pytz
import urllib.request
from flask import Flask
from threading import Thread

# -------------------------------------------------------------
# LIGHTWEIGHT FLASK SERVER (KEEPS FREE RENDER WEB SERVICE ALIVE)
# -------------------------------------------------------------
app = Flask('')

@app.route('/')
def home():
    return "StockTickerBot is online and healthy!"

def run_flask():
    # Render binds web services to port 10000 or PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# Start Flask web server in a separate background thread
Thread(target=run_flask, daemon=True).start()

# -------------------------------------------------------------
# DISCORD BOT SETUP
# -------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize Groq Client via Environment Variable
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global Scanner Settings
scanner_config = {
    "channel_id": None,
    "watchlist": [],                # Populated dynamically with S&P 500
    "vol_multiplier": 1.5,          # Default: 1.5x Avg Volume
    "timeframe": "15m",             # Default: 15m intraday bars
    "enabled": False
}

# -------------------------------------------------------------
# DYNAMIC S&P 500 TICKER FETCHING
# -------------------------------------------------------------
def get_sp500_tickers():
    """Fetches current S&P 500 ticker list from Wikipedia with browser request headers."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response)
            df = tables[0]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            print(f"✅ Successfully loaded {len(tickers)} S&P 500 tickers for volume scanning.")
            return tickers
    except Exception as e:
        print(f"⚠️ Error fetching S&P 500 list: {e}. Falling back to default list.")
        return ["AAPL", "NVDA", "TSLA", "META", "AMD", "AMZN", "MSFT", "GOOGL", "SPY", "QQQ"]

# -------------------------------------------------------------
# BLACK-SCHOLES GAMMA ENGINE
# -------------------------------------------------------------
def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_gex_and_flip(stock, current_price):
    try:
        expirations = stock.options
        if not expirations:
            return {"net_gex": "N/A", "flip": "N/A", "call_wall": "N/A", "put_wall": "N/A"}

        exps_to_use = expirations[:2]
        all_calls, all_puts = [], []

        for exp in exps_to_use:
            opt = stock.option_chain(exp)
            d2e = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 1)
            T = d2e / 365.0

            c, p = opt.calls.copy(), opt.puts.copy()
            c['T'], p['T'] = T, T
            all_calls.append(c)
            all_puts.append(p)

        calls = pd.concat(all_calls, ignore_index=True)
        puts = pd.concat(all_puts, ignore_index=True)
        r = 0.045

        calls['gamma'] = calls.apply(lambda row: bs_gamma(current_price, row['strike'], row['T'], r, row['impliedVolatility']), axis=1)
        puts['gamma'] = puts.apply(lambda row: bs_gamma(current_price, row['strike'], row['T'], r, row['impliedVolatility']), axis=1)

        calls['gex'] = calls['gamma'] * calls['openInterest'].fillna(0) * (current_price ** 2) * 0.01
        puts['gex'] = -puts['gamma'] * puts['openInterest'].fillna(0) * (current_price ** 2) * 0.01

        call_gex = calls.groupby('strike')['gex'].sum()
        put_gex = puts.groupby('strike')['gex'].sum()

        strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
        df_gex = pd.DataFrame({'strike': strikes}).set_index('strike')
        df_gex['call_gex'] = call_gex
        df_gex['put_gex'] = put_gex
        df_gex = df_gex.fillna(0)
        df_gex['net_gex'] = df_gex['call_gex'] + df_gex['put_gex']

        total_net_gex = df_gex['net_gex'].sum() / 1_000_000
        call_wall_strike = df_gex['call_gex'].idxmax()
        put_wall_strike = df_gex['put_gex'].abs().idxmax()

        zero_crossings = df_gex[np.sign(df_gex['net_gex']).diff() != 0]
        if not zero_crossings.empty:
            flip_strike = zero_crossings.index[np.abs(zero_crossings.index - current_price).argmin()]
        else:
            flip_strike = current_price

        gex_sign = "+" if total_net_gex >= 0 else ""
        
        return {
            "net_gex": f"{gex_sign}${total_net_gex:,.2f}M",
            "flip": f"${flip_strike:.2f}",
            "call_wall": f"${call_wall_strike:.2f}",
            "put_wall": f"${put_wall_strike:.2f}"
        }
    except Exception:
        return {"net_gex": "N/A", "flip": "N/A", "call_wall": "N/A", "put_wall": "N/A"}

# -------------------------------------------------------------
# FULL S&P 500 LIQUIDITY SCANNER
# -------------------------------------------------------------
def run_sp500_scanner(tickers, multiplier, interval):
    results = []
    
    try:
        data = yf.download(tickers, period="10d", interval=interval, group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Error downloading bulk data: {e}")
        return results

    for ticker in tickers:
        try:
            if ticker not in data or data[ticker].empty:
                continue

            df = data[ticker].dropna()
            if len(df) < 10:
                continue

            current_bar = df.iloc[-1]
            past_10_avg_vol = df['Volume'].iloc[-11:-1].mean()
            
            if past_10_avg_vol == 0 or np.isnan(past_10_avg_vol):
                continue
                
            current_vol = current_bar['Volume']
            ratio = current_vol / past_10_avg_vol

            open_price = current_bar['Open']
            close_price = current_bar['Close']
            pct_change = ((close_price - open_price) / open_price) * 100

            if ratio >= multiplier:
                stock_obj = yf.Ticker(ticker)
                
                # Calculate 5-Day Percent Change
                five_day_hist = stock_obj.history(period="5d")
                if len(five_day_hist) >= 2:
                    five_day_start = five_day_hist['Close'].iloc[0]
                    five_day_pct = ((close_price - five_day_start) / five_day_start) * 100
                    five_day_sign = "+" if five_day_pct >= 0 else ""
                    five_day_str = f"{five_day_sign}{five_day_pct:.2f}%"
                else:
                    five_day_str = "N/A"

                gex_metrics = calculate_gex_and_flip(stock_obj, close_price)

                results.append({
                    "ticker": ticker,
                    "price": close_price,
                    "pct_change": pct_change,
                    "five_day_pct": five_day_str,
                    "volume": current_vol,
                    "avg_vol": past_10_avg_vol,
                    "ratio": ratio,
                    "net_gex": gex_metrics["net_gex"],
                    "flip": gex_metrics["flip"],
                    "call_wall": gex_metrics["call_wall"],
                    "put_wall": gex_metrics["put_wall"]
                })
        except Exception:
            continue

    results.sort(key=lambda x: x['ratio'], reverse=True)
    return results


def build_scanner_embed(results, multiplier, interval, total_scanned):
    """Builds individual vertical cards stacked 1-by-1 for top relative volume movers."""
    embed = discord.Embed(
        title="🚨 **S&P 500 Intraday Volume Spike Scanner**",
        description=f"Scanned **{total_scanned} S&P 500 Stocks**  •  Filter: **≥{multiplier}x Avg Volume** ({interval})",
        color=0x00FF7F if results else 0x5865F2
    )

    if not results:
        embed.add_field(
            name="📊 **Scan Results**",
            value=f"```yaml\nStatus: No S&P 500 tickers currently exceed {multiplier}x average volume on {interval}.\n```",
            inline=False
        )
    else:
        for r in results[:5]: # Top 5 triggered stocks stacked vertically
            sign = "+" if r["pct_change"] >= 0 else ""
            
            card_content = (
                f"```yaml\n"
                f"• Price       : ${r['price']:.2f}\n"
                f"• Day Change  : {sign}{r['pct_change']:.2f}%\n"
                f"• 5-Day Change: {r['five_day_pct']}\n"
                f"• Volume Surge: {r['ratio']:.2f}x Avg ({r['volume']:,} shares)\n"
                f"• Net GEX     : {r['net_gex']}\n"
                f"• Gamma Flip  : {r['flip']}\n"
                f"• Call Wall   : {r['call_wall']}\n"
                f"• Put Wall    : {r['put_wall']}\n"
                f"```"
            )
            
            embed.add_field(
                name=f"📌 **${r['ticker']}**  •  {r['ratio']:.1f}x Volume Spike Triggered",
                value=card_content,
                inline=False
            )

    est_time = datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S EST')
    embed.set_footer(text=f"StockTickerBot Dynamic S&P 500 Scanner • {est_time}")
    return embed


# -------------------------------------------------------------
# HOURLY BACKGROUND TASK
# -------------------------------------------------------------
@tasks.loop(hours=1)
async def hourly_market_scanner():
    if not scanner_config["enabled"] or scanner_config["channel_id"] is None:
        return

    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    
    if now.weekday() >= 5:
        return

    market_open = time(9, 30)
    market_close = time(16, 0)

    if market_open <= now.time() <= market_close:
        channel = bot.get_channel(scanner_config["channel_id"])
        if channel:
            results = await asyncio.to_thread(
                run_sp500_scanner,
                scanner_config["watchlist"],
                scanner_config["vol_multiplier"],
                scanner_config["timeframe"]
            )
            embed = build_scanner_embed(
                results,
                scanner_config["vol_multiplier"],
                scanner_config["timeframe"],
                len(scanner_config["watchlist"])
            )
            await channel.send(embed=embed)


# -------------------------------------------------------------
# BOT EVENTS & COMMAND SYNCS
# -------------------------------------------------------------
@bot.event
async def on_ready():
    # Load full S&P 500 ticker list dynamically
    scanner_config["watchlist"] = await asyncio.to_thread(get_sp500_tickers)

    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} slash command(s).")
    except Exception as e:
        print(f"❌ Sync error: {e}")
    
    if not hourly_market_scanner.is_running():
        hourly_market_scanner.start()

    print(f'🤖 Bot online as {bot.user.name}')


# -------------------------------------------------------------
# SLASH COMMAND: /scanner_config
# -------------------------------------------------------------
@bot.tree.command(name="scanner_config", description="Configure hourly volume scanner settings across S&P 500")
@app_commands.describe(
    multiplier="Volume multiplier relative to past 10-day avg",
    timeframe="Candle timeframe interval",
    enable="Enable or disable hourly automated posts"
)
@app_commands.choices(
    multiplier=[
        app_commands.Choice(name="0.5x Avg Volume", value=0.5),
        app_commands.Choice(name="1.0x Avg Volume", value=1.0),
        app_commands.Choice(name="1.5x Avg Volume", value=1.5),
        app_commands.Choice(name="2.0x Avg Volume", value=2.0),
    ],
    timeframe=[
        app_commands.Choice(name="Daily (1D)", value="1d"),
        app_commands.Choice(name="1 Hour (1H)", value="1h"),
        app_commands.Choice(name="15 Minutes (15M)", value="15m"),
        app_commands.Choice(name="5 Minutes (5M)", value="5m"),
    ]
)
async def config_scanner(
    interaction: discord.Interaction,
    multiplier: float = None,
    timeframe: str = None,
    enable: bool = None
):
    await interaction.response.defer()

    if multiplier is not None:
        scanner_config["vol_multiplier"] = multiplier
    if timeframe is not None:
        scanner_config["timeframe"] = timeframe
    if enable is not None:
        scanner_config["enabled"] = enable
    
    scanner_config["channel_id"] = interaction.channel_id
    status_str = "ENABLED 🟢" if scanner_config["enabled"] else "DISABLED 🔴"

    embed = discord.Embed(
        title="⚙️ **S&P 500 Volume & Gamma Scanner Settings**",
        color=0x3498db
    )
    
    config_box = (
        f"```yaml\n"
        f"Status        : {status_str}\n"
        f"Multiplier    : {scanner_config['vol_multiplier']}x Avg Volume\n"
        f"Timeframe     : {scanner_config['timeframe']}\n"
        f"Target        : #{interaction.channel.name}\n"
        f"Scope         : {len(scanner_config['watchlist'])} S&P 500 Tickers\n"
        f"```"
    )
    embed.add_field(name="📋 **Current Settings**", value=config_box, inline=False)

    embed.set_footer(
        text=f"Updated by {interaction.user.display_name}",
        icon_url=interaction.user.display_avatar.url
    )

    await interaction.followup.send(embed=embed)


# -------------------------------------------------------------
# SLASH COMMAND: /scan
# -------------------------------------------------------------
@bot.tree.command(name="scan", description="Run an instant volume surge scan across all S&P 500 stocks")
async def manual_scan(interaction: discord.Interaction):
    await interaction.response.defer()

    results = await asyncio.to_thread(
        run_sp500_scanner,
        scanner_config["watchlist"],
        scanner_config["vol_multiplier"],
        scanner_config["timeframe"]
    )
    embed = build_scanner_embed(
        results,
        scanner_config["vol_multiplier"],
        scanner_config["timeframe"],
        len(scanner_config["watchlist"])
    )
    await interaction.followup.send(embed=embed)


# -------------------------------------------------------------
# SLASH COMMAND: /stock ticker:[SYMBOL]
# -------------------------------------------------------------
def fetch_stock_and_gex_data(ticker_clean: str):
    stock = yf.Ticker(ticker_clean)
    info = stock.fast_info
    
    current_price = info.last_price
    prev_close = info.previous_close
    
    if current_price is None or prev_close is None:
        return None

    change = current_price - prev_close
    pct_change = (change / prev_close) * 100
    color = 0x00FF7F if change >= 0 else 0xFF4500
    sign = "+" if change >= 0 else ""

    hist = stock.history(period="5d")
    if len(hist) >= 2:
        five_day_start = hist['Close'].iloc[0]
        five_day_change = current_price - five_day_start
        five_day_pct = (five_day_change / five_day_start) * 100
        five_day_sign = "+" if five_day_change >= 0 else ""
        five_day_str = f"{five_day_sign}${five_day_change:.2f} ({five_day_sign}{five_day_pct:.2f}%)"
    else:
        five_day_str = "N/A"

    gex_data = calculate_gex_and_flip(stock, current_price)

    return {
        "current_price": current_price,
        "prev_close": prev_close,
        "change": change,
        "pct_change": pct_change,
        "sign": sign,
        "color": color,
        "five_day_str": five_day_str,
        "net_gex_str": gex_data["net_gex"],
        "flip_level_str": gex_data["flip"],
        "call_wall_str": gex_data["call_wall"],
        "put_wall_str": gex_data["put_wall"]
    }

@bot.tree.command(name="stock", description="Get high-level price metrics and options gamma analytics")
@app_commands.describe(ticker="The stock ticker symbol (e.g. META, NVDA, AAPL)")
async def stock_slash(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker_clean = ticker.upper()
    
    try:
        data = await asyncio.to_thread(fetch_stock_and_gex_data, ticker_clean)
        if data is None:
            await interaction.followup.send(f"⚠️ Could not fetch data for **${ticker_clean}**.")
            return

        embed = discord.Embed(
            title=f"🏛️ **${ticker_clean}**  •  Market & Gamma Overview",
            color=data["color"]
        )
        embed.set_thumbnail(url=f"https://logo.clearbit.com/{ticker_clean.lower()}.com")

        price_box = (
            f"```yaml\n"
            f"Current Price : ${data['current_price']:.2f}\n"
            f"Day Change    : {data['sign']}${data['change']:.2f} ({data['sign']}{data['pct_change']:.2f}%)\n"
            f"5-Day Change  : {data['five_day_str']}\n"
            f"Prev. Close   : ${data['prev_close']:.2f}\n"
            f"```"
        )
        embed.add_field(name="📊 **Price Action**", value=price_box, inline=False)

        gamma_box = (
            f"```json\n"
            f"Net GEX     : \"{data['net_gex_str']}\"\n"
            f"Gamma Flip  : \"{data['flip_level_str']}\"\n"
            f"Call Wall   : \"{data['call_wall_str']}\"\n"
            f"Put Wall    : \"{data['put_wall_str']}\"\n"
            f"```"
        )
        embed.add_field(name="⚡ **Gamma Analytics (Near-Term)**", value=gamma_box, inline=False)

        embed.set_footer(
            text=f"Requested by {interaction.user.display_name} • Source: Yahoo Finance Options Surface",
            icon_url=interaction.user.display_avatar.url
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Error fetching stock data: {str(e)}")

# -------------------------------------------------------------
# SLASH COMMAND: /ask prompt:[QUESTION] (GROQ API CLOUD ENGINE)
# -------------------------------------------------------------
@bot.tree.command(name="ask", description="Ask the AI financial analyst a question")
@app_commands.describe(prompt="Your question for the AI")
async def ask_slash(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()

    try:
        def generate_ai():
            completion = groq_client.chat.completions.create(
                model="qwen-2.5-32b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst inside a Discord server. Keep answers direct, well-structured with clear bullet points, concise, and under 150 words."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300
            )
            return completion.choices[0].message.content

        ai_reply = await asyncio.to_thread(generate_ai)

        embed = discord.Embed(
            title="🧠 **AI Financial Analysis**",
            color=0x5865F2
        )
        embed.add_field(name="❓ **Prompt**", value=f"> *{prompt}*", inline=False)
        embed.add_field(name="💬 **Insight**", value=ai_reply, inline=False)

        embed.set_footer(
            text=f"Requested by {interaction.user.display_name} • Powered by Groq AI",
            icon_url=interaction.user.display_avatar.url
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Groq AI Error: {str(e)}")

# -------------------------------------------------------------
# START BOT VIA DISCORD_TOKEN ENVIRONMENT VARIABLE
# -------------------------------------------------------------
TOKEN = os.environ.get("DISCORD_TOKEN")
bot.run(TOKEN)
