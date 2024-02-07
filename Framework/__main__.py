
from Framework.GameManager import GameManager
from Framework.Tournament import Tournament

from Framework.ExampleBots.HumanPlayer import HumanPlayer
from Framework.ExampleBots.RandomBot import RandomBot


def example_tournament() -> None:
    print("Example of a chess tournament")
    
    # List of Competitors
    human = HumanPlayer()
    random = RandomBot()
    
    competitors = [human, random]
    
    print("Initialized competitors")
    print(f"Competitors: {[bot.name for bot in competitors]}")
    
    rounds = 10
    
    print(f"Creating a tournament with {rounds} rounds")
    # Create a tournament
    tournament = Tournament(competitors, rounds)
    
    print("Running the tournament")
    
    results = tournament.run()
    
    print("Tournament complete")
    print(results)
    
    # investigate the results
    human_wins = results.total_results[human].wins
    human_losses = results.total_results[human].losses
    
    print(f"Human wins: {human_wins}, Human losses: {human_losses}")
    
    human_vs_random = results.competition_results[(0, 1)]
    
    human_vs_random_wins = human_vs_random[human].wins
    human_vs_random_losses = human_vs_random[human].losses
    
    print(f"Human vs Random wins: {human_vs_random_wins}, Human vs Random losses: {human_vs_random_losses}")
        
    
def example_match() -> None:
    print("Example of a chess match")
    
    # Initialize competitors
    human = HumanPlayer()
    random = RandomBot()
    
    print("Initialized competitors")
    print(f"Competitors: {human.name} vs {random.name}")
    
    # Initialize the game manager
    game_manager = GameManager(human, random)
    
    print("Starting the game")
    
    # Play the game
    result, thinking_time = game_manager.play_game()
    
    print("Game complete")
    
    # Print the result
    print(f"Result: {result.result}")
    
    # Print the thinking time
    print(f"Thinking Time: {thinking_time.white} seconds for {human.name}, {thinking_time.black} seconds for {random.name}")


if __name__ == "__main__":
    # example_tournament()
    example_match()