#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRVM actions table scraper with scheduler.
- Scrapes https://www.brvm.org/fr/cours-actions/0
- Keeps the table exactly as displayed
- Adds a Date_heure column using the "Dernière mise à jour" banner
- Appends to a CSV, skipping if Date_heure already exists
- Runs on a recurring schedule (user-defined minutes)
Requirements:
    pip install pandas playwright apscheduler
    playwright install
Usage:
    python brvm_scraper_scheduler.py --interval 5 --csv brvm_cours_actions.csv
"""
import os
import sys
import asyncio
import argparse
from datetime import datetime
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from playwright.async_api import async_playwright

URL = "https://www.brvm.org/fr/cours-actions/0"

def ensure_csv_schema(path: str):
    """If CSV exists, do nothing. Otherwise create an empty file on first save."""
    # We'll create it on first successful scrape; no-op here.
    return

async def scrape_once() -> pd.DataFrame:
    """Scrape the BRVM actions table once and return a DataFrame with Date_heure column added."""
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        await page.goto(URL, timeout=90_000, wait_until="networkidle")

        # Wait for main table (has <table> in the central content)
        await page.wait_for_selector("table", timeout=60_000)

        # Grab the 'Dernière mise à jour' banner text
        # We search for an element containing that phrase
        maj_locator = page.locator("text=Dernière mise à jour")
        await maj_locator.first.wait_for(timeout=30_000)
        maj_text = await maj_locator.first.inner_text()

        # Extract the date-time part after the first colon, keep as-is
        # Example: "Dernière mise à jour : Mercredi, 24 septembre, 2025 - 14:03"
        if ":" in maj_text:
            date_heure = maj_text.split(":", 1)[-1].strip()
        else:
            # Fallback to current time if format changes unexpectedly
            date_heure = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Extract the first table HTML on the page (main table of quotes)
        html = await page.inner_html("table")
        await browser.close()

    # Convert HTML table to DataFrame
    dfs = pd.read_html(html)
    if not dfs:
        raise RuntimeError("Aucune table n'a été trouvée dans le HTML récupéré.")
    df = dfs[0]

    # Add Date_heure column at the end
    df["Date_heure"] = date_heure
    return df

def save_data(new_df: pd.DataFrame, csv_path: str):
    """Append to CSV if this Date_heure is not already present."""
    date_tag = str(new_df["Date_heure"].iloc[0])
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
        except Exception:
            # If file exists but unreadable, back it up and recreate
            bak = csv_path + ".bak"
            os.rename(csv_path, bak)
            print(f"[WARN] CSV corrompu. Sauvegarde -> {bak}. Recréation du fichier.", flush=True)
            old_df = None

        if old_df is not None and "Date_heure" in old_df.columns:
            if date_tag in set(old_df["Date_heure"].astype(str).unique()):
                print(f"[SKIP] {date_tag} déjà présent. Pas d'ajout.", flush=True)
                return

            final_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            final_df = new_df
    else:
        final_df = new_df

    final_df.to_csv(csv_path, index=False)
    print(f"[OK] Ajouté: {date_tag} ({len(new_df)} lignes)", flush=True)

def job(csv_path: str):
    """Job wrapper to be scheduled."""
    try:
        df = asyncio.run(scrape_once())
        save_data(df, csv_path)
    except Exception as e:
        print(f"[ERREUR] {e}", file=sys.stderr, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Scraper BRVM Actions avec planification.")
    parser.add_argument("--interval", type=int, default=5,
                        help="Intervalle en minutes entre deux scrapes (défaut: 5)")
    parser.add_argument("--csv", type=str, default="brvm_cours_actions.csv",
                        help="Chemin du fichier CSV de sortie (défaut: brvm_cours_actions.csv)")
    args = parser.parse_args()

    ensure_csv_schema(args.csv)
    print(f"--- BRVM Scraper démarré ---\nURL: {URL}\nCSV: {args.csv}\nIntervalle: {args.interval} min", flush=True)

    scheduler = BlockingScheduler(timezone="UTC")  # Le site affiche l'heure locale; nous n'altérons pas la chaîne
    trigger = IntervalTrigger(minutes=args.interval)

    # Exécuter une première fois immédiatement
    job(args.csv)

    # Puis planifier en continu
    scheduler.add_job(job, trigger, args=[args.csv], id="brvm_scrape_job", max_instances=1, coalesce=True)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Arrêt demandé. Bye.", flush=True)

if __name__ == "__main__":
    main()
