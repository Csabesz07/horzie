import math
import pandas as pd
from collections import defaultdict

def probability(rating1, rating2):
    return 1.0 / (1 + math.pow(10, (rating1 - rating2) / 400.0))

# player1 & player2: the current elo for each player
# K is a constant
# Outcome: 1 for Player A win, 0 for Player B win, 0.5 for draw
def elo_rating(player1elo, player2elo, K, outcome):
    Pb = probability(player1elo, player2elo)
    Pa = probability(player2elo, player1elo)

    player1elo = player1elo + K * (outcome - Pa)
    player2elo = player2elo + K * ((1 - outcome) - Pb)

    return player1elo, player2elo

def calc_elos(df: pd.DataFrame, K: float = 30.0, horse_col: str = "horse_name", place_col: str = "place", elo_col: str = "elo") -> pd.DataFrame:
    df = df.copy()

    if elo_col not in df.columns:
        df[elo_col] = 2000.0
    df[elo_col] = df[elo_col].astype(float)
    df[elo_col] = float("nan")

    # Current Elo per horse across races
    current_elo = {}

    # Buffer of row indices belonging to the current race
    race_idx = []

    def process_race(indices):
        if not indices:
            return

        # Capture race participants (index, horse_id/name, place)
        participants = []
        for idx in indices:
            horse = df.at[idx, horse_col]
            place = df.at[idx, place_col]

            # Initialize current elo if first time seen
            if horse not in current_elo:
                # If df already has an elo seed for this row, use it; else 2000
                seed = df.at[idx, elo_col]
                current_elo[horse] = float(seed) if pd.notna(seed) else 2000.0

            # Write PRE-RACE elo into df[elo_col] for this row
            df.at[idx, elo_col] = current_elo[horse]

            participants.append((idx, horse, int(place)))

        # Sort by finishing place (1 is best). This makes outcome logic straightforward.
        participants.sort(key=lambda t: t[2])

        # Sum Elo deltas per horse (simultaneous update)
        delta = defaultdict(float)

        # Pairwise comparisons
        n = len(participants)
        for i in range(n):
            idx_i, horse_i, place_i = participants[i]
            elo_i = current_elo[horse_i]

            for j in range(i + 1, n):
                idx_j, horse_j, place_j = participants[j]
                elo_j = current_elo[horse_j]

                # i has better place than j because of sorting -> i "wins"
                new_i, new_j = elo_rating(elo_i, elo_j, K=K, outcome=1.0)

                delta[horse_i] += (new_i - elo_i)
                delta[horse_j] += (new_j - elo_j)

        # Apply summed deltas simultaneously
        for idx, horse, _ in participants:
            current_elo[horse] = current_elo[horse] + delta[horse]
            df.at[idx, elo_col] = current_elo[horse]

    # Iterate rows in order; start a new race when we see place==1
    for idx, row in df.iterrows():
        place = row[place_col]

        # Start of a new race detected
        if int(place) == 1:
            # If we already have buffered rows, process the previous race
            if race_idx:
                process_race(race_idx)
                race_idx = []

        race_idx.append(idx)

    # Process last buffered race
    process_race(race_idx)

    return df


def calc_riders(riders):
    for i, p1 in riders:    
        if len(riders) == i + 1:
            break

        for y, p2 in riders[i+1:]:
            results = elo_rating(player1elo=p1['elo'], player2elo=p2['elo'], K=30, outcome=1)
            p1['elo'] = results[0]['elo']
            p2['elo'] = results[1]['elo']