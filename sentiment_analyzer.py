"""
sentiment_analyzer.py - Alpha Vantage NEWS_SENTIMENT (Free plan) + aggregation

- Uses Alpha Vantage NEWS_SENTIMENT endpoint
- Caches JSON responses to avoid repeat calls
- Parses per-article and ticker sentiment (Alpha Vantage fields are best-effort)
- Aggregates daily signals and adds rolling windows + momentum
- Exposes FinancialSentimentAnalyzer.run_pipeline(symbol, start_date, end_date)
- Saves CSV to data/sentiment/{SYMBOL}_sentiment.csv

Requires:
    pip install requests pandas numpy python-dateutil
Environment:
    ALPHA_VANTAGE_API_KEY must be set (or pass api_key to class)
"""
from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional
from datetime import datetime, timezone
from dateutil import parser as dateparser

import requests
import pandas as pd
import numpy as np

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = "data/sentiment_cache"
OUTPUT_DIR = "data/sentiment"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FinancialSentimentAnalyzer:
    """
    Alpha Vantage NEWS_SENTIMENT wrapper.

    Usage:
        analyzer = FinancialSentimentAnalyzer(api_key="XXX")
        df = analyzer.run_pipeline("AAPL", "2023-01-01", "2024-01-01")
    """

    def __init__(self, api_key: Optional[str] = None, max_articles: int = 500):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY or pass api_key.")
        self.max_articles = int(max_articles)

    def _cache_path(self, symbol: str, start: str, end: str) -> str:
        safe_symbol = symbol.upper()
        return os.path.join(CACHE_DIR, f"{safe_symbol}__{start}__{end}.json")

    def fetch_feed(self, symbol: str, start_date: str, end_date: str, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch feed from Alpha Vantage and cache the JSON.
        start_date and end_date are ISO strings 'YYYY-MM-DD'
        """
        cache_file = self._cache_path(symbol, start_date, end_date)
        if os.path.exists(cache_file) and not force_refresh:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                feed = payload.get("feed", []) or payload.get("items", []) or []
                return feed[: self.max_articles]
            except Exception:
                pass

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": start_date,
            "time_to": end_date,
            "apikey": self.api_key
        }

        try:
            r = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()

            # Rate-limit / note handling
            if isinstance(payload, dict) and ("Note" in payload or "Information" in payload):
                # save what we got, but warn
                print("⚠️ Alpha Vantage returned Note/Information (possible rate limit).")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            feed = payload.get("feed", []) or payload.get("items", []) or []
            return feed[: self.max_articles]
        except Exception as e:
            print(f"❌ Alpha Vantage fetch failed: {e}")
            return []

    @staticmethod
    def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
        if not ts_str:
            return None
        try:
            dt = dateparser.parse(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    @staticmethod
    def _av_label_to_numeric(label: Optional[str]) -> float:
        """Map common labels to numeric [-1,1]"""
        if not label:
            return 0.0
        mapping = {
            "positive": 1.0,
            "somewhat-positive": 0.5,
            "neutral": 0.0,
            "somewhat-negative": -0.5,
            "negative": -1.0
        }
        return float(mapping.get(label.lower(), 0.0))

    def _parse_item(self, item: Dict) -> Optional[Dict]:
        """
        Parse one feed item. AlphaVantage items vary in keys; handle robustly.
        Returns standardized dict:
            {date(datetime), title, summary, av_label, av_score, per_ticker_scores(dict)}
        """
        try:
            title = item.get("title") or item.get("headline") or ""
            summary = item.get("summary") or item.get("text") or item.get("summary_text") or ""
            published = item.get("time_published") or item.get("published_at") or item.get("datetime") or item.get("date")
            dt = self._parse_timestamp(published) or datetime.now(timezone.utc)

            av_label = item.get("overall_sentiment_label") or item.get("overall_sentiment") or None
            av_score = item.get("overall_sentiment_score", None)
            try:
                if av_score is not None:
                    av_score = float(av_score)
                    av_score = max(-1.0, min(1.0, av_score))
            except Exception:
                av_score = None

            # ticker_sentiment may exist as list
            per_ticker = {}
            tlist = item.get("ticker_sentiment") or item.get("tickers") or []
            if isinstance(tlist, list):
                for t in tlist:
                    try:
                        tk = t.get("ticker") or t.get("symbol")
                        score = t.get("ticker_sentiment_score") or t.get("sentiment_score")
                        if score is not None:
                            score = float(score)
                            score = max(-1.0, min(1.0, score))
                        per_ticker[tk] = score
                    except Exception:
                        continue

            return {
                "title": title,
                "summary": summary,
                "time_published": dt,
                "av_label": av_label,
                "av_score": av_score,
                "per_ticker": per_ticker
            }
        except Exception:
            return None

    def calculate_daily_sentiment(self, feed: List[Dict], symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Convert raw feed list into a daily aggregated DataFrame.
        Columns:
            Sentiment_Score, Article_Count, AlphaVantage_Score_mean,
            Sentiment_Roll_3d, Sentiment_Roll_7d, Sentiment_Roll_14d, Sentiment_Momentum_1d
        """
        if not feed:
            return pd.DataFrame()

        parsed = []
        for it in feed:
            p = self._parse_item(it)
            if p is None:
                continue
            # If symbol present in per_ticker, prefer that score
            if symbol:
                st = p["per_ticker"].get(symbol.upper())
                if st is not None:
                    p["av_score"] = st
            parsed.append(p)

        if not parsed:
            return pd.DataFrame()

        rows = []
        for p in parsed:
            date = p["time_published"].astimezone(timezone.utc).date()
            # choose av_score numeric if present, else map av_label
            av_score = p["av_score"]
            if av_score is None and p["av_label"]:
                av_score = self._av_label_to_numeric(p["av_label"])
            # fallback 0
            av_score = 0.0 if av_score is None else float(av_score)
            # small heuristic: combined score = av_score (AlphaVantage is reasonably good)
            combined = av_score
            rows.append({"date": pd.to_datetime(date), "combined_score": combined, "av_score": av_score})

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()

        agg = df.groupby("date").agg(
            Sentiment_Score=("combined_score", "mean"),
            Article_Count=("combined_score", "count"),
            AlphaVantage_Score=("av_score", "mean")
        ).reset_index().set_index("date").sort_index()

        # create continuous date index
        start = agg.index.min()
        end = agg.index.max()
        all_days = pd.date_range(start=start, end=end, freq="D")
        agg = agg.reindex(all_days)
        agg.index.name = "Date"

        # fill
        agg["Sentiment_Score"] = agg["Sentiment_Score"].ffill().fillna(0.0)
        agg["AlphaVantage_Score"] = agg["AlphaVantage_Score"].ffill().fillna(0.0)
        agg["Article_Count"] = agg["Article_Count"].fillna(0).astype(int)

        # Rolling features
        agg["Sentiment_Roll_3d"] = agg["Sentiment_Score"].rolling(3, min_periods=1).mean()
        agg["Sentiment_Roll_7d"] = agg["Sentiment_Score"].rolling(7, min_periods=1).mean()
        agg["Sentiment_Roll_14d"] = agg["Sentiment_Score"].rolling(14, min_periods=1).mean()
        agg["Sentiment_Momentum_1d"] = agg["Sentiment_Score"].diff().fillna(0.0)

        # final columns
        result = agg[[
            "Sentiment_Score", "Article_Count", "AlphaVantage_Score",
            "Sentiment_Roll_3d", "Sentiment_Roll_7d", "Sentiment_Roll_14d", "Sentiment_Momentum_1d"
        ]].copy()

        # save location note
        return result

    def save_sentiment(self, sentiment_df: pd.DataFrame, symbol: str) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{symbol.upper()}_sentiment.csv")
        sentiment_df.to_csv(path, index=True)
        print(f"✓ Sentiment saved to {path}")
        return path

    def run_pipeline(self, symbol: str, start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
        feed = self.fetch_feed(symbol, start_date, end_date, force_refresh=force_refresh)
        print(f"Fetched {len(feed)} items for {symbol} from Alpha Vantage.")
        daily = self.calculate_daily_sentiment(feed, symbol=symbol)
        if daily is not None and not daily.empty:
            self.save_sentiment(daily, symbol)
        return daily
