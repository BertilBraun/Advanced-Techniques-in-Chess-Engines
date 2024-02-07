
from Framework.GameManager import GameManager
from Framework.Tournament import Tournament

from Framework.ExampleBots.HumanBot import HumanBot
from Framework.ExampleBots.RandomBot import RandomBot


def example_tournament() -> None:
    print("Example of a chess tournament")
    
    # List of Competitors
    human = HumanBot()
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
    
    # investigate the results
    human_wins = results[human].wins
    human_losses = results[human].losses
    
    print(f"Human wins: {human_wins}, Human losses: {human_losses}")
    
    
def example_match() -> None:
    print("Example of a chess match")
    
    # Initialize competitors
    human = HumanBot()
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