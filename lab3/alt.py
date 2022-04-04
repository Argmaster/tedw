from collections import OrderedDict
from inspect import isclass
from typing import Dict, List, Tuple, Type


class RecordMeta(type):
    def __new__(
        cls, name: str, bases: List[type], attributes: Dict[str, type]
    ):
        annotations = attributes.get("__annotations__")
        fields: Dict[str, type] = OrderedDict()
        if annotations is not None:
            cls._inherit_fields(bases, fields)
            cls._own_fields(annotations, fields)
        class_ob = super().__new__(cls, name, bases, attributes)
        class_ob.__fields__ = fields
        print(fields)
        return class_ob

    @staticmethod
    def _inherit_fields(
        bases: Tuple[Type["RecordBase"]],
        fields: Dict[str, type],
    ):
        for base in bases:
            if issubclass(base, RecordBase):
                fields.update(get_fields(base))

    @staticmethod
    def _own_fields(
        annotations: dict,
        fields: Dict[str, type],
    ) -> dict:
        for name, field in annotations.items():
            fields[name] = field


def get_fields(ob: "RecordBase") -> dict:
    if isclass(ob):
        return ob.__fields__
    else:
        return ob.__class__.__fields__


class RecordBase(metaclass=RecordMeta):
    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(
            {key: value for key, value in zip(self.__fields__, args)}
        )
        self.__dict__.update(
            {
                key: value
                for key, value in kwargs.items()
                if key in self.__fields__
            }
        )
        for key in self.__fields__:
            if key not in self.__dict__:
                raise ValueError(f"Missing value for {key}.")

    def __str__(self):
        fields = ", ".join(
            f"{key}={self.__dict__[key]}" for key in self.__fields__
        )
        return f"{self.__class__.__qualname__}({fields})"

    __repr__ = __str__

    def to_dict(self):
        return {key: self.__dict__[key] for key in self.__fields__}

    def to_tuple(self):
        return tuple(self.__dict__[key] for key in self.__fields__)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, RecordBase):
            return self.to_dict() == __o.to_dict()
        else:
            return self == __o

    def __hash__(self):
        return hash((self.__class__, self.to_tuple()))


class PumpkinSeed(RecordBase):
    a: int
    b: float


class Other(PumpkinSeed):
    c: int


class Cluster:

    ob_list: List[RecordBase]
