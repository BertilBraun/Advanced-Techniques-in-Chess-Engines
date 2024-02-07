
from Framework.Tournament import Tournament
from Framework.ExampleBots.RandomBot import RandomBot
from HandcraftedChessBot.HandcraftedBotV1 import HandcraftedBotV1
from HandcraftedChessBot.HandcraftedBotV2 import HandcraftedBotV2


if __name__ == "__main__":
    print("Handcrafted chess tournament")
        
    competitors = [
        RandomBot(), 
        HandcraftedBotV1(), 
        HandcraftedBotV2(),
    ]
    
    print("Initialized competitors")
    tournament = Tournament(competitors, rounds=5)
    
    print("Initialized tournament")
    print(tournament)
    
    print("Running the tournament")
    
    results = tournament.run()
    
    print("Tournament complete")
    print(results)
    