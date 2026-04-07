# Prediction vs actual XI — comparison report

**Generated:** 2026-04-07T04:39:08.678113+00:00

**Data sources:** `parsers.router.parse_scorecard` (IPLT20 match URLs), 
`squad_fetch.fetch_squad_for_slug` (current official squads), `predictor.run_prediction`.

**Caveats:** Squads are **current** IPLT20 listings; XIs for past matches may include 
players not on today’s page. Impact Player **actual** selections are **not** parsed from 
scorecards in this pipeline — only **model-predicted** impact order is shown.

**XI / bowling comparison keys:** Overlap, missed/extra, and bowling-usage sets use ``player_registry.audit_player_identity_key`` (aliases / history keys), then ``learner.normalize_player_key`` fallback — audit layer only; prediction unchanged.

**Run mode:** SQLite ``match_results`` preloaded payloads (last 3 rows with team_a/team_b).

---

## Summary

- Matches requested: **3**
- Matches audited (parsed + predicted): **3**

## A. Top recurring mismatch patterns (heuristic tags)

| Tag | Count |
| --- | --- |
| repair overreach | 1 |

## B. Players frequently mispredicted

### Most often **missed** (in actual XI, not in predicted XI)

| Player | Miss count |
| --- | --- |
| Harsh Dubey | 1 |
| Shivang Kumar | 1 |
| DA Payne | 1 |
| A Raghuvanshi | 1 |
| Kartik Tyagi | 1 |
| MD Choudhary | 1 |
| Prince Yadav | 1 |
| P Nissanka | 1 |
| V Nigam | 1 |
| N Wadhera | 1 |

### Most often **false positives** (predicted XI, not in actual XI)

| Player | Extra count |
| --- | --- |
| Harshal Patel | 1 |
| Travis Head | 1 |
| Pat Cummins | 1 |
| Blessing Muzarabani | 1 |
| Sarthak Ranjan | 1 |
| Matthew Breetzke | 1 |
| Avesh Khan | 1 |
| Mitchell Starc | 1 |
| Abishek Porel | 1 |
| Harpreet Brar | 1 |

## C. Teams where XI overlap is lowest (this run)

| Team (canonical) | Min overlap /11 | Mean overlap /11 | Innings |
| --- | --- | --- | --- |
| Sunrisers Hyderabad | 8 | 8.00 | 1 |
| Kolkata Knight Riders | 9 | 9.00 | 1 |
| Lucknow Super Giants | 9 | 9.00 | 1 |
| Delhi Capitals | 9 | 9.00 | 1 |
| Punjab Kings | 10 | 10.00 | 1 |
| Gujarat Titans | 11 | 11.00 | 1 |

## D. Conditions where the model struggled (low overlap, this run)

Teams with **correct_picks < 8** (arbitrary threshold for this report):

— None under threshold, or no data.

For each match, see **Conditions used** (venue, weather snapshot, toss unknown) in the per-match section.

---

## Per-match detail

### Match 1

- **URL:** `cricsheet://all/1527679`
- **Payload source:** `sqlite_preload`
- **Date / venue / result:** 2026-04-02 · Eden Gardens, Kolkata · Sunrisers Hyderabad (runs 65)
- **Conditions:** venue=Eden Gardens, Kolkata · match_time=2026-04-02T19:30:00 · toss=unknown (audit default)

#### Kolkata Knight Riders

- **Predicted XI:** Ramandeep Singh, Anukul Roy, Rinku Singh, Ajinkya Rahane, Finn Allen, Sunil Narine, Vaibhav Arora, Varun Chakaravarthy, Blessing Muzarabani, Cameron Green, Sarthak Ranjan
- **Actual XI (scorecard):** FH Allen, AM Rahane, A Raghuvanshi, C Green, RK Singh, AS Roy, Ramandeep Singh, SP Narine, Kartik Tyagi, VG Arora, CV Varun
- **Overlap count (registry-aware audit keys):** 9 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 8
- **Missed (actual not predicted):** A Raghuvanshi, Kartik Tyagi
- **Extra (predicted not actual):** Blessing Muzarabani, Sarthak Ranjan
- **Batting order:** top-3 positional matches=0 · openers(2)=0 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=0
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=6 · predicted XI classified as bowling options=7 · bowlers who bowled and were in predicted XI=5
  - **Bowled but not in predicted XI:** Kartik Tyagi
  - **Predicted XI bowling options with no recorded overs:** Cameron Green, Sarthak Ranjan
- **Impact subs (model):** Angkrish Raghuvanshi, Umran Malik, Rahul Tripathi, Kartik Tyagi, Tim Seifert
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._
- **Mismatch notes (heuristic):** repair overreach

#### Sunrisers Hyderabad

- **Predicted XI:** Harshal Patel, Aniket Verma, Abhishek Sharma, Eshan Malinga, Jaydev Unadkat, Travis Head, Ishan Kishan, Nitish Kumar Reddy, Heinrich Klaasen, Salil Arora, Pat Cummins
- **Actual XI (scorecard):** E Malinga, Abhishek Sharma, Ishan Kishan, H Klaasen, Aniket Verma, Nithish Kumar Reddy, S Arora, Harsh Dubey, Shivang Kumar, JD Unadkat, DA Payne
- **Overlap count (registry-aware audit keys):** 8 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 5
- **Missed (actual not predicted):** Harsh Dubey, Shivang Kumar, DA Payne
- **Extra (predicted not actual):** Harshal Patel, Travis Head, Pat Cummins
- **Batting order:** top-3 positional matches=2 · openers(2)=1 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=0
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=7 · predicted XI classified as bowling options=5 · bowlers who bowled and were in predicted XI=4
  - **Bowled but not in predicted XI:** DA Payne, Harsh Dubey, Shivang Kumar
  - **Predicted XI bowling options with no recorded overs:** Harshal Patel, Pat Cummins
- **Impact subs (model):** Zeeshan Ansari, David Payne, Kamindu Mendis, Liam Livingstone, Brydon Carse
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._


### Match 2

- **URL:** `cricsheet://all/1527678`
- **Payload source:** `sqlite_preload`
- **Date / venue / result:** 2026-04-01 · Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow · Delhi Capitals (wickets 6)
- **Conditions:** venue=BRSABV Ekana Stadium, Lucknow · match_time=2026-04-01T19:30:00 · toss=unknown (audit default)

#### Delhi Capitals

- **Predicted XI:** KL Rahul, Kuldeep Yadav, Axar Patel, Tristan Stubbs, David Miller, Mitchell Starc, Sameer Rizvi, Mukesh Kumar, Lungisani Ngidi, Nitish Rana, Abishek Porel
- **Actual XI (scorecard):** Sameer Rizvi, KL Rahul, P Nissanka, N Rana, AR Patel, T Stubbs, V Nigam, DA Miller, L Ngidi, Kuldeep Yadav, Mukesh Kumar
- **Overlap count (registry-aware audit keys):** 9 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 5
- **Missed (actual not predicted):** P Nissanka, V Nigam
- **Extra (predicted not actual):** Mitchell Starc, Abishek Porel
- **Batting order:** top-3 positional matches=1 · openers(2)=1 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=0
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=6 · predicted XI classified as bowling options=6 · bowlers who bowled and were in predicted XI=4
  - **Bowled but not in predicted XI:** T Natarajan, V Nigam
  - **Predicted XI bowling options with no recorded overs:** Mitchell Starc, Sameer Rizvi
- **Impact subs (model):** Karun Nair, T. Natarajan, Dushmantha Chameera, Vipraj Nigam, Pathum Nissanka
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._

#### Lucknow Super Giants

- **Predicted XI:** Rishabh Pant, Anrich Nortje, Nicholas Pooran, Mohsin Khan, Matthew Breetzke, Ayush Badoni, Mohammad Shami, Abdul Samad, Avesh Khan, Aiden Markram, Shahbaz Ahamad
- **Actual XI (scorecard):** Shahbaz Ahmed, RR Pant, AK Markram, A Badoni, N Pooran, Abdul Samad, MD Choudhary, Mohammed Shami, A Nortje, Mohsin Khan, Prince Yadav
- **Overlap count (registry-aware audit keys):** 9 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 7
- **Missed (actual not predicted):** MD Choudhary, Prince Yadav
- **Extra (predicted not actual):** Matthew Breetzke, Avesh Khan
- **Batting order:** top-3 positional matches=0 · openers(2)=0 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=1
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=7 · predicted XI classified as bowling options=6 · bowlers who bowled and were in predicted XI=6
  - **Bowled but not in predicted XI:** Prince Yadav
  - **Predicted XI bowling options with no recorded overs:** Avesh Khan
- **Impact subs (model):** M. Siddharth, Wanindu Hasaranga, Arjun Tendulkar, Prince Yadav, Mitchell Marsh
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._


### Match 3

- **URL:** `cricsheet://all/1527677`
- **Payload source:** `sqlite_preload`
- **Date / venue / result:** 2026-03-31 · Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh · Punjab Kings (wickets 3)
- **Conditions:** venue=Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh · match_time=2026-03-31T19:30:00 · toss=unknown (audit default)

#### Gujarat Titans

- **Predicted XI:** Rahul Tewatia, Ashok Sharma, Mohammed Siraj, Prasidh Krishna, Rashid Khan, Glenn Phillips, Shubman Gill, Sai Sudharsan, Jos Buttler, Kagiso Rabada, Washington Sundar
- **Actual XI (scorecard):** M Prasidh Krishna, B Sai Sudharsan, Shubman Gill, JC Buttler, GD Phillips, Washington Sundar, R Tewatia, Rashid Khan, K Rabada, Ashok Sharma, Mohammed Siraj
- **Overlap count (registry-aware audit keys):** 11 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 6
- **Missed (actual not predicted):** —
- **Extra (predicted not actual):** —
- **Batting order:** top-3 positional matches=0 · openers(2)=0 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=0
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=6 · predicted XI classified as bowling options=8 · bowlers who bowled and were in predicted XI=6
  - **Predicted XI bowling options with no recorded overs:** Rahul Tewatia, Glenn Phillips
- **Impact subs (model):** Ishant Sharma, Sai Kishore, Luke Wood, Shahrukh Khan, Anuj Rawat
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._

#### Punjab Kings

- **Predicted XI:** Arshdeep Singh, Vyshak Vijaykumar, Shreyas Iyer, Priyansh Arya, Shashank Singh, Harpreet Brar, Xavier Bartlett, Marco Jansen, Cooper Connolly, Marcus Stoinis, Prabhsimran Singh
- **Actual XI (scorecard):** Priyansh Arya, P Simran Singh, C Connolly, SS Iyer, N Wadhera, Shashank Singh, MP Stoinis, M Jansen, XC Bartlett, Vijaykumar Vyshak, Arshdeep Singh
- **Overlap count (registry-aware audit keys):** 10 / 11
  - **Registry-bridged pairs** (same player, different scorecard vs squad spelling): 7
- **Missed (actual not predicted):** N Wadhera
- **Extra (predicted not actual):** Harpreet Brar
- **Batting order:** top-3 positional matches=1 · openers(2)=1 · middle slots 4–7 (set overlap)=1 · lower 8–11 (set)=0
- **Bowling (scorecard vs predicted XI):** distinct bowlers who bowled=5 · predicted XI classified as bowling options=8 · bowlers who bowled and were in predicted XI=4
  - **Bowled but not in predicted XI:** YS Chahal
  - **Predicted XI bowling options with no recorded overs:** Priyansh Arya, Harpreet Brar, Cooper Connolly, Marcus Stoinis
- **Impact subs (model):** Nehal Wadhera, Yash Thakur, Azmatullah Omarzai, Musheer Khan, Suryansh Shedge
  - **Actual (scorecard):** not in parser schema
  - _IPL Impact Player is not a structured field in the current HTML/JSON parser; use match commentary or official supersub lists for ground truth._


---

## Regenerate

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/generate_prediction_vs_actual_report.py
```
