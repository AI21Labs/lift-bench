from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin, cfg, dataclass_json


@dataclass
class PolymorphicDataclassEntry:
    type_field_name: str
    child_classes_str_to_type: dict[str, type] = field(default_factory=dict)
    child_classes_type_to_str: dict[type, str] = field(default_factory=dict)


_polymorphic_dataclasses: dict[type, PolymorphicDataclassEntry] = {}


def find_polymorphic_baseclass(cls):
    if cls in _polymorphic_dataclasses:
        return cls
    for base_cls in cls.__bases__[::-1]:
        polymorphic_baseclass = find_polymorphic_baseclass(base_cls)
        if polymorphic_baseclass:
            return polymorphic_baseclass
    return None


def _register_json_dataclasses_hooks(cls_to_register, cls_to_use):
    cfg.global_config.encoders[cls_to_register] = cls_to_use.to_dict
    cfg.global_config.decoders[cls_to_register] = cls_to_use.from_dict
    # Workaround until https://github.com/lidatong/dataclasses-json/issues/512 is resolved
    cfg.global_config.encoders[List[cls_to_register]] = cls_to_use.list_to_dict
    cfg.global_config.decoders[List[cls_to_register]] = cls_to_use.list_from_dict
    cfg.global_config.encoders[Optional[cls_to_register]] = cls_to_use.to_dict
    cfg.global_config.decoders[Optional[cls_to_register]] = cls_to_use.from_dict


def register_polymorphic_baseclass(cls, type_field_name: str):
    _polymorphic_dataclasses[cls] = PolymorphicDataclassEntry(type_field_name)
    _register_json_dataclasses_hooks(cls, cls)


def register_polymorphic_childclass(cls, type_field_value: str):
    parent_cls = find_polymorphic_baseclass(cls)
    _polymorphic_dataclasses[parent_cls].child_classes_str_to_type[type_field_value] = cls
    _polymorphic_dataclasses[parent_cls].child_classes_type_to_str[cls] = type_field_value
    _register_json_dataclasses_hooks(cls, parent_cls)


class PolymorphicDataClassJsonMixin(DataClassJsonMixin):
    @staticmethod
    def list_to_dict(list_instance, encode_json=False):
        return [PolymorphicDataClassJsonMixin.to_dict(item, encode_json=encode_json) for item in list_instance]

    @classmethod
    def list_from_dict(cls, list_instance, infer_missing=False):
        return [cls.from_dict(item, infer_missing=infer_missing) for item in list_instance]

    def to_dict(self, encode_json=False):
        ret_val = DataClassJsonMixin.to_dict(self, encode_json=encode_json)
        if ret_val is None:
            return None
        parent_cls = find_polymorphic_baseclass(type(self))
        entry = _polymorphic_dataclasses[parent_cls]
        ret_val[entry.type_field_name] = entry.child_classes_type_to_str[type(self)]
        return ret_val

    @classmethod
    def from_dict(cls, data, infer_missing=False):
        if cls not in _polymorphic_dataclasses:
            cls = find_polymorphic_baseclass(cls)
        entry = _polymorphic_dataclasses[cls]
        type_field_name = entry.type_field_name
        data_cloned = data.copy()
        type_field = data_cloned.pop(type_field_name)
        child_class = entry.child_classes_str_to_type[type_field]
        if hasattr(child_class, "preprocess_dict"):
            child_class.preprocess_dict(data_cloned)
        return DataClassJsonMixin.from_dict.__func__(child_class, data_cloned, infer_missing=infer_missing)

    @classmethod
    def preprocess_dict(cls, data):
        """An optional function to handle data dict preprocessing such as version migration"""


def polymorphic_baseclass(_cls=None, *, type_field_name: str = None):
    def wrap(cls, *args, **kwargs):
        cls = dataclass_json(cls, *args, **kwargs)
        cls.to_dict = PolymorphicDataClassJsonMixin.to_dict
        cls.from_dict = classmethod(PolymorphicDataClassJsonMixin.from_dict.__func__)
        cls.list_to_dict = PolymorphicDataClassJsonMixin.list_to_dict
        cls.list_from_dict = classmethod(PolymorphicDataClassJsonMixin.list_from_dict.__func__)
        register_polymorphic_baseclass(cls, type_field_name or "type")
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def polymorphic_childclass(_cls=None, *, type_field_value: str = None):
    def wrap(cls, *args, **kwargs):
        cls = dataclass_json(cls, *args, **kwargs)
        cls.to_dict = PolymorphicDataClassJsonMixin.to_dict
        cls.from_dict = classmethod(PolymorphicDataClassJsonMixin.from_dict.__func__)
        cls.list_to_dict = PolymorphicDataClassJsonMixin.list_to_dict
        cls.list_from_dict = classmethod(PolymorphicDataClassJsonMixin.list_from_dict.__func__)
        register_polymorphic_childclass(cls, type_field_value or cls.__name__)
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)
