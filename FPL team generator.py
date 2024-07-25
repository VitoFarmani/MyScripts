# The script uses a genetic algorithm to find the best combination based on the score defined in the calculate_score function.
# You can modify the calculate_score for your desired score

from deap import base, creator, tools, algorithms
import random
import requests
import pandas as pd

# User Instructions:
print("""
When you run the script it asks for input, you can either enter a player's name or write 'auto'.
If you enter 'auto' the script fills the remaining empty positions based on the best overall score.
""")

bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static"
response = requests.get(bootstrap_url)

fixtures_json = response.json()
fixtures_json.keys()

teams_df = pd.DataFrame(fixtures_json['teams'])[['id', 'name', 'strength', 'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away']]
teams_df = teams_df.rename(columns = {'id' : 'team_id','name': 'team_name'})

useful_elements_df = pd.DataFrame(fixtures_json['elements'])[['team', 'id','web_name','element_type', 'now_cost','selected_by_percent', 'starts',
                                                              'total_points', 'goals_scored','assists', 'clean_sheets_per_90',
                                                              'goals_conceded_per_90', 'bonus', 'yellow_cards','red_cards','value_season']]
useful_elements_df = useful_elements_df.merge(teams_df[['team_id','team_name','strength', 'strength_attack_home',
                                                        'strength_attack_away', 'strength_defence_home','strength_defence_away']], left_on='team', right_on='team_id', how='left')
useful_elements_df.drop(['id', 'team_id', 'team'], axis=1, inplace=True)
useful_elements_df['element_type'] = useful_elements_df['element_type'].map({1:'Goalkeeper', 2:'Defender',3:'Midfielder',4:'Forward'})
useful_elements_df['now_cost'] = useful_elements_df['now_cost']/10
useful_elements_df['value_season'] = pd.to_numeric(useful_elements_df['value_season'])

# Normalize the relevant columns
columns_to_normalize = ['clean_sheets_per_90', 'goals_conceded_per_90', 'bonus', 'assists', 'goals_scored',
                        'yellow_cards', 'red_cards', 'strength_attack_home', 'strength_attack_away',
                        'strength_defence_home', 'strength_defence_away', 'starts', 'value_season']

for column in columns_to_normalize:
    useful_elements_df[column] = useful_elements_df[column] / useful_elements_df[column].max()

# Define weights for different metrics
def calculate_score(row):
    if row['element_type'] in ['Goalkeeper', 'Defender']:
        score = (
            row['goals_scored'] * 5 + row['assists'] * 4 +
            row['bonus'] * 2 - row['yellow_cards'] * 1 - row['red_cards'] * 2 +
            row['strength_attack_home'] * 4 + row['strength_attack_away'] * 4
            + row['goals_conceded_per_90'] * 2 + row['clean_sheets_per_90']
            + row['starts'] * 2 + row['value_season'] * 4
        )
    else:  # Midfielders and Forwards
        score = (
            row['goals_scored'] * 5 + row['assists'] * 4 +
            row['bonus'] * 2 - row['yellow_cards'] * 1 - row['red_cards'] * 2 +
            row['strength_attack_home'] * 4 + row['strength_attack_away'] * 4
            + row['starts'] * 2 + row['value_season'] * 4
        )
    return round(score, 2)

# Calculate score for each player
useful_elements_df['score'] = useful_elements_df.apply(calculate_score, axis=1)
useful_elements_df['score_to_cost'] = round(useful_elements_df['score'] / useful_elements_df['now_cost'], 2)

# Set up GA parameters
population_size = 100
generations = 50
mutation_probability = 0.2
crossover_probability = 0.5

# Define player positions and number of players in each position
positions = {
    'Goalkeeper': 2,
    'Defender': 5,
    'Midfielder': 5,
    'Forward': 3
}

# Create a list of players grouped by position
players_by_position = {
    'Goalkeeper': useful_elements_df[useful_elements_df['element_type'] == 'Goalkeeper'].index.tolist(),
    'Defender': useful_elements_df[useful_elements_df['element_type'] == 'Defender'].index.tolist(),
    'Midfielder': useful_elements_df[useful_elements_df['element_type'] == 'Midfielder'].index.tolist(),
    'Forward': useful_elements_df[useful_elements_df['element_type'] == 'Forward'].index.tolist()
}

# Pre-selected players
pre_selected_players = []


def get_minimum_cost_for_remaining_positions(remaining_positions):
    min_cost = 0
    for position, count in remaining_positions.items():
        if count > 0:
            min_cost += useful_elements_df.loc[players_by_position[position], 'now_cost'].nsmallest(count).sum()
    return min_cost

def select_player_manually():
    remaining_positions = positions.copy()
    remaining_budget = 100.0
    selected_players = []
    team_count = {team: 0 for team in useful_elements_df['team_name'].unique()}

    while True:
        print("\nRemaining positions needed:")
        for pos, count in remaining_positions.items():
            print(f"{pos}: {count}")
        print(f"\nRemaining budget: {remaining_budget}")

        if len(selected_players) >= 15:
            break

        choice = input("Enter the name of the player to select or 'auto' to let the script select automatically: ").strip()

        if choice.lower() == 'auto':
            break

        chosen_player = useful_elements_df[useful_elements_df['web_name'].str.lower() == choice.lower()]
        if not chosen_player.empty:
            player_idx = chosen_player.index[0]
            player_cost = chosen_player.iloc[0]['now_cost']
            player_position = chosen_player.iloc[0]['element_type']
            if player_idx in pre_selected_players:
                print("This player has already been selected. Please choose a different player.")
                continue
            min_cost_for_remaining = get_minimum_cost_for_remaining_positions(remaining_positions)
            if remaining_budget - player_cost < min_cost_for_remaining:
                print(f"Selecting {chosen_player.iloc[0]['web_name']} will leave insufficient budget for remaining positions. Please choose a different player.")
                continue
            if remaining_positions[player_position] > 0 and remaining_budget >= player_cost and team_count[chosen_player.iloc[0]['team_name']] < 3:
                pre_selected_players.append(player_idx)
                selected_players.append(player_idx)
                remaining_positions[player_position] -= 1
                remaining_budget -= player_cost
                team_count[chosen_player.iloc[0]['team_name']] += 1
                # Remove the chosen player from all lists
                for pos in positions.keys():
                    if player_idx in players_by_position[pos]:
                        players_by_position[pos].remove(player_idx)
            else:
                print("Invalid selection: Check position requirements, budget, or team limit.")
        else:
            print("Invalid choice, please try again.")
    return selected_players, remaining_positions, remaining_budget, team_count

selected_players, remaining_positions, remaining_budget, team_count = select_player_manually()

# Update positions based on pre-selected players
for player_idx in pre_selected_players:
    position = useful_elements_df.loc[player_idx, 'element_type']
    positions[position] -= 1

# Set up DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_individual():
    individual = pre_selected_players.copy()
    for position, count in positions.items():
        if count > 0:
            potential_players = [p for p in players_by_position[position] if p not in pre_selected_players]
            while count > 0 and potential_players:
                chosen_player = random.choice(potential_players)
                individual.append(chosen_player)
                potential_players.remove(chosen_player)
                count -= 1
    random.shuffle(individual)
    return creator.Individual(individual)

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    team = useful_elements_df.loc[individual]
    total_cost = team['now_cost'].sum()
    total_score = team['score'].sum()
    if total_cost > 100:
        return -1e6,  # Invalid teams have a very low fitness
    # Check if the team meets position requirements
    position_counts = team['element_type'].value_counts()
    if (position_counts.get('Goalkeeper', 0) != 2 or
        position_counts.get('Defender', 0) != 5 or
        position_counts.get('Midfielder', 0) != 5 or
        position_counts.get('Forward', 0) != 3):
        return -1e6,  # Invalid teams have a very low fitness
    # Check if the team meets the team limit requirement
    team_counts = team['team_name'].value_counts()
    if any(count > 3 for count in team_counts):
        return -1e6,  # Invalid teams have a very low fitness
    return total_score,

toolbox.register("mate", tools.cxTwoPoint)

def mutate_individual(individual):
    non_pre_selected = [p for p in individual if p not in pre_selected_players]  # Remove pre-selected players
    for _ in range(random.randint(1, 3)):
        pos = random.choice(list(positions.keys()))
        pos_indices = [i for i in range(len(non_pre_selected)) if useful_elements_df.loc[non_pre_selected[i], 'element_type'] == pos]
        if pos_indices:
            idx = random.choice(pos_indices)
            new_player = random.choice(players_by_position[pos])
            while new_player in non_pre_selected or new_player in pre_selected_players:
                new_player = random.choice(players_by_position[pos])
            non_pre_selected[idx] = new_player
    non_pre_selected.extend(pre_selected_players)  # Re-add pre-selected players
    return creator.Individual(non_pre_selected),

toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Create population
population = toolbox.population(n=population_size)

print("Please wait while the algorithm selects the best team...")

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generations, verbose=False)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
best_team = useful_elements_df.loc[best_individual]

# Check if the total cost exceeds 100
if best_team['now_cost'].sum() > 100:
    print("Not enough budget to form a valid team with the selected players.")
else:
    # Display the selected players and calculate overall score
    total_score = best_team['score'].sum()
    total_cost = best_team['now_cost'].sum()
    best_team.rename(columns={'web_name':'Name',
                                'element_type': 'Position',
                                'team_name': 'Team',
                                'now_cost' :'Cost'}, inplace=True)
    print("\nSelected players:")
    print(best_team[['Name', 'Position', 'Team', 'Cost']])
    # print(f"\nTotal Score: {total_score}")
    print(f"Total Cost: {total_cost}")