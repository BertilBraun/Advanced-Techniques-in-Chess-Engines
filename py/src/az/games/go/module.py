from src.az.games.api import GameIdentifier, GameModuleRegistration


GO_GAME_MODULE = GameModuleRegistration(
    identifier=GameIdentifier.GO,
    display_name='Go',
    payload_schema_name='go-training-payload',
)
