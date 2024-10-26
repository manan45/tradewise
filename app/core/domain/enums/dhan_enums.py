from enum import Enum

class ExchangeSegment(Enum):
    IDX_I = 0
    NSE_EQ = 1
    NSE_FNO = 2
    NSE_CURRENCY = 3
    BSE_EQ = 4
    MCX_COMM = 5
    BSE_CURRENCY = 7
    BSE_FNO = 8

class ProductType(Enum):
    CNC = "CNC"
    INTRADAY = "INTRADAY"
    MARGIN = "MARGIN"
    CO = "CO"
    BO = "BO"

class Instrument(Enum):
    INDEX = "INDEX"
    FUTIDX = "FUTIDX"
    OPTIDX = "OPTIDX"
    EQUITY = "EQUITY"
    FUTSTK = "FUTSTK"
    OPTSTK = "OPTSTK"

class ExpiryCode(Enum):
    CURRENT = 0
    NEXT = 1
    FAR = 2
