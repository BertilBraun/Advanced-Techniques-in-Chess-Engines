import argparse
from Framework.Tournament import Tournament
from Framework.ExampleBots.RandomBot import RandomBot
from HandcraftedChessBot.HandcraftedBotV1 import HandcraftedBotV1
from HandcraftedChessBot.HandcraftedBotV2 import HandcraftedBotV2
from HandcraftedChessBot.HandcraftedBotV3 import HandcraftedBotV3
from HandcraftedChessBot.HandcraftedBotV4 import HandcraftedBotV4

ALL_COMPETITORS = [
    RandomBot(),
    HandcraftedBotV1(),
    HandcraftedBotV2(),
    HandcraftedBotV3(),
    HandcraftedBotV4(),
]


def run_complete_tournament() -> None:
    tournament = Tournament(ALL_COMPETITORS, rounds=5)

    print('Initialized tournament')
    print(tournament)

    print('Running the tournament')

    results = tournament.run()

    print('Tournament complete')
    print(results)


def compete_two_latest() -> None:
    tournament = Tournament(ALL_COMPETITORS[-2:], rounds=5)

    print('Initialized competition between the two latest bots')
    print(tournament)

    print('Running the competition')

    results = tournament.run()

    print('Competition complete')
    print(results)


def play_most_recent_bot() -> None:
    from Framework.GameManager import GameManager
    from Framework.ExampleBots.HumanPlayer import HumanPlayer

    latest_bot = ALL_COMPETITORS[-1]
    game_manager = GameManager(HumanPlayer(), latest_bot)

    result, thinking_time = game_manager.play_game()

    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a chess tournament')
    parser.add_argument('--complete', action='store_true', help='Run a complete tournament')
    parser.add_argument('--compete', action='store_true', help='Compete the two latest bots')
    parser.add_argument('--play', action='store_true', help='Play against the most recent bot')
    args = parser.parse_args()

    assert args.complete + args.compete + args.play == 1, 'Exactly one of --complete, --compete, --play must be set'

    if args.complete:
        run_complete_tournament()
    elif args.compete:
        compete_two_latest()
    else:
        play_most_recent_bot()
