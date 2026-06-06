from nba_api.stats.static import teams

nba_teams = teams.get_teams()

print(f"Found {len(nba_teams)} NBA teams:\n")

for team in nba_teams:
    print(f"{team['id']} | {team['abbreviation']} | {team['full_name']}")