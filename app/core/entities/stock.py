# app/core/entities/stock.py

class Stock:
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price

    def update_price(self, new_price: float):
        self.price = new_priceclass Stock:
    def __init__(self, symbol, name, price):
        self.symbol = symbol
        self.name = name
        self.price = price

    def update_price(self, new_price):
        self.price = new_price
