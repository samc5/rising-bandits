import pandas as pd

d2020 = pd.read_csv('2020plays.csv')
d2021 = pd.read_csv('2021plays.csv')
d2022 = pd.read_csv('2022plays.csv')
d2023 = pd.read_csv('2023plays.csv')
d2024 = pd.read_csv('2024plays.csv')

merged = pd.concat([d2020, d2021, d2022, d2023, d2024], ignore_index=True)
# merged.to_csv('merged_plays.csv', index=False)

# Delete everything except for relevant column (gameid, playerid, hit or walk indicator (max of single, double, triple, homer, walk))
merged["reward"] = merged[["single", "double", "triple", "hr", "walk"]].max(axis=1)
merged = merged[["gid", "batter", "pitcher", "reward", "bat_f"]]

# turn gameid into just chronological aprt (remove first 3 chars)
merged["gid"] = merged["gid"].str[3:]
# construct r_t(i) for all players whose careers started 2020 and after (this can be found elsewhere)
biofile = pd.read_csv("biofile.csv")
biofile = biofile[['PLAYERID', "PLAY.DEBUT"]]
biofile = biofile[biofile["PLAY.DEBUT"].str[-4:] >= "2020"]
biofile.columns = ["batter", "debut"]
youths = pd.merge(merged, biofile,on="batter")
youths['trial'] = youths.groupby((youths['batter'] != youths['batter'].shift(1)).cumsum()).cumcount()+1
trial_counts = youths['batter'].value_counts()
eligible_batters = trial_counts[trial_counts >= 300].index
youths = youths[youths['batter'].isin(eligible_batters)]
# remove all players with <300 trials


youths.to_csv("youths.csv")


# combine on id with player debuts, keep only those who debuted in 2020 and beyond to make sure we're looking at beginning of careers