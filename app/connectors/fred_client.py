"""FRED + CFTC COT + CBOE VIX term-structure adapter (MacroDataProvider)."""
from __future__ import annotations

from datetime import date

import httpx
import polars as pl


class FredClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()

    async def get_series(
        self,
        series_id: str,
        start: date,
        end: date,
    ) -> list[tuple[date, float]]:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
        client = self._get_client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        rows = []
        for obs in resp.json().get("observations", []):
            if obs["value"] == ".":
                continue
            rows.append((date.fromisoformat(obs["date"]), float(obs["value"])))
        return rows

    async def get_cot_report(
        self,
        market: str,
        as_of: date,
    ) -> dict[str, float]:
        # CFTC Disaggregated report (publicly available ZIP)
        year = as_of.year
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"
        client = self._get_client()
        resp = await client.get(url)
        resp.raise_for_status()
        import io
        import zipfile
        import csv
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = [n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".txt")]
        if not names:
            return {}
        with zf.open(names[0]) as f:
            reader = csv.DictReader(io.TextIOWrapper(f))
            for row in reader:
                if market.lower() in row.get("Market_and_Exchange_Names", "").lower():
                    report_date = row.get("Report_Date_as_YYYY-MM-DD", "")
                    if report_date and date.fromisoformat(report_date) <= as_of:
                        return {
                            "comm_long": float(row.get("Prod_Merc_Positions_Long_ALL", 0)),
                            "comm_short": float(row.get("Prod_Merc_Positions_Short_ALL", 0)),
                            "noncomm_long": float(row.get("M_Money_Positions_Long_ALL", 0)),
                            "noncomm_short": float(row.get("M_Money_Positions_Short_ALL", 0)),
                        }
        return {}

    async def get_vix_term_structure(self, as_of: date) -> dict[str, float]:
        # CBOE historical VIX data
        vix_urls = {
            "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
            "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
            "VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
        }
        result: dict[str, float] = {}
        client = self._get_client()
        import csv
        import io
        for name, url in vix_urls.items():
            resp = await client.get(url)
            resp.raise_for_status()
            reader = csv.DictReader(io.StringIO(resp.text))
            last_val: float | None = None
            for row in reader:
                row_date_str = row.get("DATE", row.get("Date", ""))
                if not row_date_str:
                    continue
                try:
                    row_date = date.fromisoformat(row_date_str)
                except ValueError:
                    continue
                if row_date <= as_of:
                    close_str = row.get("CLOSE", row.get("Close", ""))
                    if close_str:
                        try:
                            last_val = float(close_str)
                        except ValueError:
                            pass
            if last_val is None:
                raise ValueError(f"No VIX data found for {name} as of {as_of}")
            result[name] = last_val
        if "VIX" in result and "VIX3M" in result:
            result["VIX_V3M_RATIO"] = result["VIX"] / max(result["VIX3M"], 0.001)
        return result
