#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape horse-race results from 2003 to 2025 and save to CSV.

This script looks for JSON-like JS blocks such as:
  races_table_divs["..."] = { ... "participants":[...], "description":"..." };

It is resilient to missing days, retries transient failures, and writes rows
incrementally so you can resume without losing progress.
"""

import csv
import json
import random
import re
import sys
import time
from datetime import date, timedelta
import requests

URL_TEMPLATE = "https://mla.kincsempark.hu/results/gallop/{date}"

OUT_CSV = "races_2003_2025.csv"
USER_AGENT = "RaceScraper/1.0 (+contact: your-email)"
TIMEOUT = 25
MAX_RETRIES = 3
SLEEP_BETWEEN = (1.2, 2.6)  # polite delays

FIELDNAMES = [
    "date","race_of_the_day","start_time","race_name","distance",
    "versenydij","versenykiiras","temperature_c","race_time",
    "place","program_number","horse_name","age","sex","color",
    "jockey","carried_weight","trainer","stable","form","dividend",
    "time","sire","dam"
]

re_results_date = re.compile(r'var\s+results_date\s*=\s*"([^"]+)"')
re_blocks = re.compile(r'races_table_divs\["[^"]+"\]\s*=\s*({.*?})\s*;', re.DOTALL)

def parse_temp_and_racetime(desc: str):
    if not desc:
        return None, None
    # Temperature: "Hőmérséklet: 6 °C" (handle diacritics or fallback)
    m_temp = re.search(r'H[őo]m[ée]rs[ée]klet:\s*([+-]?\d+(?:[.,]\d+)?)', desc)
    temp = m_temp.group(1).replace(",", ".") if m_temp else None
    # Race time: "Lefutási idő/Race Time: 1:21.1" (various spellings)
    m_rt = re.search(r'(?:Lefut[áa]si id[őo]|Race Time)\s*:\s*([0-9]:[0-9]{2}\.[0-9])', desc)
    race_time = m_rt.group(1) if m_rt else None
    return temp, race_time

def extract_header_near(html: str, blk: str):
    """
    Try to pull Start time, Race name, Versenydíj, Táv, Versenykiírás
    from a 'race-head-items' block near the JSON.
    """
    header = {"start_time": None, "race_name": None, "versenydij": None,
              "distance": None, "versenykiiras": None}

    idx = html.find(blk)
    if idx == -1:
        return header
    head_start = html.rfind('<div class="race-head-items">', 0, idx)
    if head_start == -1:
        return header
    # Take a chunk after the header to include its inner items
    snippet = html[head_start: head_start + 6000]

    def grab(label_regex):
        m = re.search(label_regex + r'.*?<span>([^<]+)</span>', snippet, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    header["start_time"]    = grab(r'>\s*Start\s*<')
    header["race_name"]     = grab(r'>\s*Futam neve\s*<')
    header["versenydij"]    = grab(r'>\s*Versenyd[íi]j\s*<')
    header["distance"]      = grab(r'>\s*T[áa]v\s*<')
    header["versenykiiras"] = grab(r'>\s*Versenyki[íi]r[áa]s\s*<')
    return header

def clean_json_like(s: str):
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s

def fetch(session: requests.Session, url: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return None  # no data this day
            r.raise_for_status()
            # trust UTF-8 unless server says otherwise
            r.encoding = r.encoding or "utf-8"
            return r.text
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[!] {url} failed after {attempt} tries: {e}", file=sys.stderr)
                return None
            sleep_for = random.uniform(1.0, 2.0) * attempt
            time.sleep(sleep_for)

def iterate_dates(y0=2018, y1=2025):
    d = date(y0, 1, 1)
    end = date(y1, 12, 31)
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)

def main():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # initialize CSV (write header if empty/new)
    try:
        open(OUT_CSV, "x", encoding="utf-8", newline="").close()
        need_header = True
    except FileExistsError:
        need_header = False

    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()

        for dstr in iterate_dates():
            url = URL_TEMPLATE.format(date=dstr)
            html = fetch(session, url)
            if not html:
                # no page or failed; skip politely
                time.sleep(random.uniform(*SLEEP_BETWEEN))
                continue

            # quick existence check
            if 'races_table_divs' not in html:
                time.sleep(random.uniform(*SLEEP_BETWEEN))
                continue

            # page-level date (if present)
            mdate = re_results_date.search(html)
            page_date = mdate.group(1) if mdate else dstr

            # find all race JSON blocks
            blocks = re_blocks.findall(html)
            if not blocks:
                time.sleep(random.uniform(*SLEEP_BETWEEN))
                continue

            for blk in blocks:
                head = extract_header_near(html, blk)

                try:
                    data = json.loads(clean_json_like(blk))
                except Exception:
                    # give up on this block if not parseable
                    continue

                temp_c, race_time = parse_temp_and_racetime(data.get("description", ""))

                base = {
                    "date": page_date,
                    "race_of_the_day": data.get("daily"),
                    "start_time": head["start_time"],
                    "race_name": head["race_name"],
                    "distance": head["distance"] or data.get("distance"),
                    "versenydij": head["versenydij"] or data.get("prize"),
                    "versenykiiras": head["versenykiiras"],
                    "temperature_c": temp_c,
                    "race_time": race_time,
                }

                for p in data.get("participants", []) or []:
                    row = dict(base)
                    row.update({
                        "place": p.get("rank"),
                        "program_number": p.get("number"),
                        "horse_name": p.get("name"),
                        "age": p.get("age"),
                        "sex": p.get("sex"),
                        "color": p.get("color"),
                        "jockey": p.get("driver_jockey"),
                        "carried_weight": p.get("weight"),
                        "trainer": p.get("trainer"),
                        "stable": p.get("stable"),
                        "form": p.get("rate"),
                        "dividend": p.get("dividend"),
                        "time": p.get("time"),
                        "sire": p.get("sire"),
                        "dam": p.get("dam"),
                    })
                    writer.writerow(row)

            print(f"[{dstr}] done ({len(blocks)} races)")
            sys.stdout.flush()

            # polite delay
            time.sleep(random.uniform(*SLEEP_BETWEEN))

    print(f"Done. CSV saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
