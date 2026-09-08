from enum import Enum


class FunctionEnum(str, Enum):
    ADVANCED_FLOAT = "AdvancedFloat"
    ADVANCED_INTEGER = "AdvancedInteger"
    AMALGAMATION = "Amalgamation"
    FLOAT = "Float"
    INTEGER = "Integer"
    ORDERED_SYMBOL = "OrderedSymbol"
    STRING = "String"
    SYMBOL = "Symbol"
    TAXONOMY = "Taxonomy"
    DATE = "Date"
    INTERVAL = "Interval"
    ADVANCED_DOUBLE = "AdvancedDouble"
    DOUBLE = "Double"
