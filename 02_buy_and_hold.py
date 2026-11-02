# 02_buy_and_hold.py
# Stap 2: de benchmark meten. Wat leverde "op dag 1 SPY kopen en nooit meer
# iets doen" op, en hoeveel risico liep je daarbij?
# Uitkomst: jaarrendement (CAGR), volatiliteit, Sharpe ratio en maximale drawdown.
#
# Dit script gebruikt data/SPY.csv uit stap 1. Draai eerst 01_data_ophalen.py
# als dat bestand er niet staat.

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "SPY"
HANDELSDAGEN_PER_JAAR = 252

# ---------- 1. Koersdata inlezen ----------
data = pd.read_csv(f"data/{TICKER}.csv", index_col="Date", parse_dates=True)

# Dagrendement: (slotkoers vandaag / slotkoers gisteren) - 1.
# De eerste dag heeft geen "gisteren", dus die halen we weg.
rendement = data["Close"].pct_change().dropna()

# ---------- 2. Risicovrije rente ophalen ----------
# ^IRX is de rente op 3-maands Amerikaanse staatsleningen (T-bills),
# in procenten per jaar. Dat is wat je "zonder risico" had kunnen verdienen.
print("Risicovrije rente (^IRX) ophalen...")
rente = yf.Ticker("^IRX").history(start=rendement.index[0], auto_adjust=False)
rente.index = rente.index.tz_localize(None)

# Van procent per jaar naar fractie per dag (bijv. 2.0 -> 0.02 / 252).
rente_per_dag = rente["Close"] / 100 / HANDELSDAGEN_PER_JAAR

# De rente-data heeft soms andere dagen dan SPY (feestdagen).
# We zetten hem op dezelfde datums als SPY en vullen gaten
# met de laatst bekende waarde (ffill = "forward fill").
rente_per_dag = rente_per_dag.reindex(rendement.index).ffill().fillna(0)

# ---------- 3. Waarde van de belegging door de tijd ----------
# Begin met 1 euro. Elke dag: waarde x (1 + rendement van die dag).
# cumprod = alle factoren achter elkaar vermenigvuldigen.
waarde = (1 + rendement).cumprod()

# ---------- 4. Kengetallen ----------
# Jaarrendement (CAGR): welk vast percentage per jaar geeft dezelfde eindwaarde?
aantal_jaren = (rendement.index[-1] - rendement.index[0]).days / 365.25
totaal_rendement = waarde.iloc[-1] - 1
cagr = waarde.iloc[-1] ** (1 / aantal_jaren) - 1

# Volatiliteit: standaarddeviatie van de dagrendementen, opgeschaald naar een jaar.
volatiliteit = rendement.std() * np.sqrt(HANDELSDAGEN_PER_JAAR)

# Sharpe ratio: gemiddeld rendement boven de risicovrije rente,
# gedeeld door de schommeling daarvan. Eerst per dag, dan naar een jaar.
overrendement = rendement - rente_per_dag
sharpe = overrendement.mean() / overrendement.std() * np.sqrt(HANDELSDAGEN_PER_JAAR)

# Gemiddelde risicovrije rente over de periode (ter info).
gem_rente = rente_per_dag.mean() * HANDELSDAGEN_PER_JAAR

# Maximale drawdown: hoe ver zakte de waarde onder de hoogste stand tot dan toe?
hoogste_tot_nu = waarde.cummax()
drawdown = waarde / hoogste_tot_nu - 1
max_drawdown = drawdown.min()
dal_datum = drawdown.idxmin()
piek_datum = waarde.loc[:dal_datum].idxmax()

# Herstel: eerste dag na het dal waarop de oude piek weer werd gehaald.
na_dal = waarde.loc[dal_datum:]
hersteld = na_dal[na_dal >= waarde.loc[piek_datum]]
if len(hersteld) > 0:
    herstel_datum = hersteld.index[0].date()
else:
    herstel_datum = "nog niet hersteld"

# ---------- 5. Resultaten tonen ----------
print(f"\nBuy-and-hold {TICKER}, {rendement.index[0].date()} t/m {rendement.index[-1].date()}")
print(f"  Aantal jaren:            {aantal_jaren:.1f}")
print(f"  Totaal rendement:        {totaal_rendement:.0%}")
print(f"  Jaarrendement (CAGR):    {cagr:.2%}")
print(f"  Volatiliteit (per jaar): {volatiliteit:.2%}")
print(f"  Gem. risicovrije rente:  {gem_rente:.2%}")
print(f"  Sharpe ratio:            {sharpe:.2f}")
print(f"  Maximale drawdown:       {max_drawdown:.2%}")
print(f"    Piek:    {piek_datum.date()}")
print(f"    Dal:     {dal_datum.date()}")
print(f"    Herstel: {herstel_datum}")

# Kengetallen opslaan, zodat we straks strategieën ernaast kunnen zetten.
os.makedirs("results", exist_ok=True)
kengetallen = pd.DataFrame({
    "Buy-and-hold": {
        "CAGR": round(cagr, 4),
        "Volatiliteit": round(volatiliteit, 4),
        "Sharpe": round(sharpe, 2),
        "Max drawdown": round(max_drawdown, 4),
    }
})
kengetallen.to_csv("results/02_kengetallen_buy_and_hold.csv")
print("\nKengetallen opgeslagen in results/02_kengetallen_buy_and_hold.csv")

# ---------- 6. Grafieken ----------
fig, assen = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Boven: wat werd 1 euro waard? (log-schaal)
assen[0].plot(waarde.index, waarde)
assen[0].set_yscale("log")
assen[0].set_title(f"Buy-and-hold {TICKER}: waarde van 1 euro (log-schaal)")
assen[0].set_ylabel("Waarde")

# Onder: drawdown. Hoe ver zat je onder je hoogste stand?
assen[1].fill_between(drawdown.index, drawdown, 0, color="tab:red", alpha=0.4)
assen[1].set_title("Drawdown (afstand tot de hoogste stand tot dan toe)")
assen[1].set_ylabel("Drawdown")

plt.tight_layout()
plt.savefig(f"results/02_buy_and_hold_{TICKER}.png", dpi=150)
print(f"Grafiek opgeslagen in results/02_buy_and_hold_{TICKER}.png")
plt.show()
