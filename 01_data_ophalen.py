# 01_data_ophalen.py
# Stap 1: historische koersen van SPY (ETF op de S&P 500) ophalen,
# controleren of de data klopt, opslaan en in een grafiek zetten.

import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Instellingen ----------
TICKER = "SPY"          # ETF die de S&P 500 volgt
STARTDATUM = "2000-01-01"  # met dotcom-crash, kredietcrisis 2008 en corona 2020

# ---------- 1. Data downloaden ----------
# auto_adjust=True: koersen worden gecorrigeerd voor dividend en splits,
# zodat het rendement klopt met wat een belegger echt verdiende.
print(f"Data ophalen voor {TICKER} vanaf {STARTDATUM}...")
data = yf.Ticker(TICKER).history(start=STARTDATUM, auto_adjust=True)

# Yahoo geeft tijdzones mee aan de datums; die hebben we niet nodig.
data.index = data.index.tz_localize(None)

# We houden alleen de kolommen die we gebruiken.
data = data[["Open", "High", "Low", "Close", "Volume"]]

# ---------- 2. Controleren ----------
print("\nEerste 5 rijen:")
print(data.head())
print("\nLaatste 5 rijen:")
print(data.tail())

print(f"\nAantal handelsdagen: {len(data)}")
print(f"Van {data.index[0].date()} tot {data.index[-1].date()}")

# Ontbrekende waarden tellen (zou 0 moeten zijn)
print("\nOntbrekende waarden per kolom:")
print(data.isna().sum())

# ---------- 3. Dagrendement berekenen ----------
# Rendement vandaag = (slotkoers vandaag / slotkoers gisteren) - 1
data["Rendement"] = data["Close"].pct_change()

print("\nGrootste dagstijging en -daling:")
print(f"  Beste dag:  {data['Rendement'].idxmax().date()}  {data['Rendement'].max():.2%}")
print(f"  Slechtste dag: {data['Rendement'].idxmin().date()}  {data['Rendement'].min():.2%}")

# ---------- 4. Opslaan ----------
# Data gaat in de map data/ (die zetten we in .gitignore, niet op GitHub).
os.makedirs("data", exist_ok=True)
data.to_csv(f"data/{TICKER}.csv")
print(f"\nOpgeslagen in data/{TICKER}.csv")

# ---------- 5. Grafieken ----------
os.makedirs("results", exist_ok=True)

fig, assen = plt.subplots(2, 1, figsize=(11, 7))

# Bovenste grafiek: koers op logaritmische schaal.
# Log-schaal: een stijging van 10% ziet er altijd even groot uit,
# of de koers nu 100 of 500 is.
assen[0].plot(data.index, data["Close"])
assen[0].set_yscale("log")
assen[0].set_title(f"{TICKER} koers (gecorrigeerd voor dividend), log-schaal")
assen[0].set_ylabel("Koers ($)")

# Onderste grafiek: dagrendementen. Hier zie je wanneer de markt onrustig was.
assen[1].plot(data.index, data["Rendement"], linewidth=0.5)
assen[1].set_title("Dagrendement")
assen[1].set_ylabel("Rendement")

plt.tight_layout()
plt.savefig(f"results/01_{TICKER}_koers_en_rendement.png", dpi=150)
print(f"Grafiek opgeslagen in results/01_{TICKER}_koers_en_rendement.png")
plt.show()
