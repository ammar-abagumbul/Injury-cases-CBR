from enum import Enum


class DescriptionEnum(str, Enum):
    BOOLEAN = "Boolean"
    CONCEPT = "Concept"
    DATE = "Date"
    INTERVAL = "Interval"
    INTEGER = "Integer"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    SYMBOL = "Symbol"
