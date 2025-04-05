import AlphaZeroCpp

result = AlphaZeroCpp.decode_move(5)
print(result.__dir__())
print(result.promotion_type().__dir__())
print(result.promotion_type().value)
