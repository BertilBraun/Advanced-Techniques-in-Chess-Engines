
from Framework.Tournament import Tournament
from HandcraftedChessBot.HandcraftedBotV1 import HandcraftedBotV1
from Framework.ExampleBots.RandomBot import RandomBot


if __name__ == "__main__":
    print("Handcrafted chess tournament")
        
    competitors = [
        RandomBot(), 
        HandcraftedBotV1()
    ]
    
    print("Initialized competitors")
    tournament = Tournament(competitors, rounds=100)
    
    print("Initialized tournament")
    print(tournament)
    
    print("Running the tournament")
    
    results = tournament.run()
    
    print("Tournament complete")
    print(results)
    