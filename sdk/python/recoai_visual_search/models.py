# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = add_image_from_dict(json.loads(json_string))
#     result = common_from_dict(json.loads(json_string))
#     result = remove_collection_from_dict(json.loads(json_string))
#     result = remove_image_from_dict(json.loads(json_string))
#     result = search_image_from_dict(json.loads(json_string))
#     result = upsert_collection_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import List, Any, Optional, Dict, TypeVar, Callable, Type, cast
from enum import Enum


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


@dataclass
class ImageBytes:
    bytes: List[int]

    @staticmethod
    def from_dict(obj: Any) -> 'ImageBytes':
        assert isinstance(obj, dict)
        bytes = from_list(from_int, obj.get("bytes"))
        return ImageBytes(bytes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["bytes"] = from_list(from_int, self.bytes)
        return result


@dataclass
class ImageSource:
    image_bytes: Optional[ImageBytes] = None
    url: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ImageSource':
        assert isinstance(obj, dict)
        image_bytes = from_union([ImageBytes.from_dict, from_none], obj.get("ImageBytes"))
        url = from_union([from_str, from_none], obj.get("Url"))
        return ImageSource(image_bytes, url)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ImageBytes"] = from_union([lambda x: to_class(ImageBytes, x), from_none], self.image_bytes)
        result["Url"] = from_union([from_str, from_none], self.url)
        return result


@dataclass
class AddImage:
    collection_name: str
    id: str
    source: ImageSource

    @staticmethod
    def from_dict(obj: Any) -> 'AddImage':
        assert isinstance(obj, dict)
        collection_name = from_str(obj.get("collection_name"))
        id = from_str(obj.get("id"))
        source = ImageSource.from_dict(obj.get("source"))
        return AddImage(collection_name, id, source)

    def to_dict(self) -> dict:
        result: dict = {}
        result["collection_name"] = from_str(self.collection_name)
        result["id"] = from_str(self.id)
        result["source"] = to_class(ImageSource, self.source)
        return result


@dataclass
class RemoveCollection:
    name: str

    @staticmethod
    def from_dict(obj: Any) -> 'RemoveCollection':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        return RemoveCollection(name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        return result


@dataclass
class RemoveImage:
    id: str
    index_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'RemoveImage':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        index_name = from_str(obj.get("index_name"))
        return RemoveImage(id, index_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = from_str(self.id)
        result["index_name"] = from_str(self.index_name)
        return result


@dataclass
class SearchImage:
    collection_name: str
    n_results: int
    source: ImageSource

    @staticmethod
    def from_dict(obj: Any) -> 'SearchImage':
        assert isinstance(obj, dict)
        collection_name = from_str(obj.get("collection_name"))
        n_results = from_int(obj.get("n_results"))
        source = ImageSource.from_dict(obj.get("source"))
        return SearchImage(collection_name, n_results, source)

    def to_dict(self) -> dict:
        result: dict = {}
        result["collection_name"] = from_str(self.collection_name)
        result["n_results"] = from_int(self.n_results)
        result["source"] = to_class(ImageSource, self.source)
        return result


class ModelArchitecture(Enum):
    EFFICIENT_NET_LITE4 = "EfficientNetLite4"
    MOBILE_NET_V2 = "MobileNetV2"
    RES_NET152 = "ResNet152"
    SQUEEZE_NET = "SqueezeNet"


class Channels(Enum):
    CWH = "CWH"
    WHC = "WHC"


@dataclass
class ImageSize:
    height: int
    width: int

    @staticmethod
    def from_dict(obj: Any) -> 'ImageSize':
        assert isinstance(obj, dict)
        height = from_int(obj.get("height"))
        width = from_int(obj.get("width"))
        return ImageSize(height, width)

    def to_dict(self) -> dict:
        result: dict = {}
        result["height"] = from_int(self.height)
        result["width"] = from_int(self.width)
        return result


@dataclass
class CenterCrop:
    crop_size: ImageSize

    @staticmethod
    def from_dict(obj: Any) -> 'CenterCrop':
        assert isinstance(obj, dict)
        crop_size = ImageSize.from_dict(obj.get("crop_size"))
        return CenterCrop(crop_size)

    def to_dict(self) -> dict:
        result: dict = {}
        result["crop_size"] = to_class(ImageSize, self.crop_size)
        return result


@dataclass
class Normalization:
    div: List[float]
    sub: List[float]
    zeroone: bool

    @staticmethod
    def from_dict(obj: Any) -> 'Normalization':
        assert isinstance(obj, dict)
        div = from_list(from_float, obj.get("div"))
        sub = from_list(from_float, obj.get("sub"))
        zeroone = from_bool(obj.get("zeroone"))
        return Normalization(div, sub, zeroone)

    def to_dict(self) -> dict:
        result: dict = {}
        result["div"] = from_list(to_float, self.div)
        result["sub"] = from_list(to_float, self.sub)
        result["zeroone"] = from_bool(self.zeroone)
        return result


class FilterType(Enum):
    CATMULL_ROM = "CatmullRom"
    GAUSSIAN = "Gaussian"
    LANCZOS3 = "Lanczos3"
    NEAREST = "Nearest"
    TRIANGLE = "Triangle"


@dataclass
class ResizeRGBImage:
    filter: FilterType
    image_size: ImageSize

    @staticmethod
    def from_dict(obj: Any) -> 'ResizeRGBImage':
        assert isinstance(obj, dict)
        filter = FilterType(obj.get("filter"))
        image_size = ImageSize.from_dict(obj.get("image_size"))
        return ResizeRGBImage(filter, image_size)

    def to_dict(self) -> dict:
        result: dict = {}
        result["filter"] = to_enum(FilterType, self.filter)
        result["image_size"] = to_class(ImageSize, self.image_size)
        return result


@dataclass
class ResizeRGBImageAspectRatio:
    filter: FilterType
    image_size: ImageSize
    scale: float

    @staticmethod
    def from_dict(obj: Any) -> 'ResizeRGBImageAspectRatio':
        assert isinstance(obj, dict)
        filter = FilterType(obj.get("filter"))
        image_size = ImageSize.from_dict(obj.get("image_size"))
        scale = from_float(obj.get("scale"))
        return ResizeRGBImageAspectRatio(filter, image_size, scale)

    def to_dict(self) -> dict:
        result: dict = {}
        result["filter"] = to_enum(FilterType, self.filter)
        result["image_size"] = to_class(ImageSize, self.image_size)
        result["scale"] = to_float(self.scale)
        return result


@dataclass
class Transpose:
    axes: List[int]

    @staticmethod
    def from_dict(obj: Any) -> 'Transpose':
        assert isinstance(obj, dict)
        axes = from_list(from_int, obj.get("axes"))
        return Transpose(axes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["axes"] = from_list(from_int, self.axes)
        return result


@dataclass
class ImageTransform:
    resize_rgb_image: Optional[ResizeRGBImage] = None
    resize_rgb_image_aspect_ratio: Optional[ResizeRGBImageAspectRatio] = None
    center_crop: Optional[CenterCrop] = None
    normalization: Optional[Normalization] = None
    transpose: Optional[Transpose] = None
    to_array: Optional[Dict[str, Any]] = None
    to_tensor: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ImageTransform':
        assert isinstance(obj, dict)
        resize_rgb_image = from_union([ResizeRGBImage.from_dict, from_none], obj.get("ResizeRGBImage"))
        resize_rgb_image_aspect_ratio = from_union([ResizeRGBImageAspectRatio.from_dict, from_none], obj.get("ResizeRGBImageAspectRatio"))
        center_crop = from_union([CenterCrop.from_dict, from_none], obj.get("CenterCrop"))
        normalization = from_union([Normalization.from_dict, from_none], obj.get("Normalization"))
        transpose = from_union([Transpose.from_dict, from_none], obj.get("Transpose"))
        to_array = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("ToArray"))
        to_tensor = from_union([lambda x: from_dict(lambda x: x, x), from_none], obj.get("ToTensor"))
        return ImageTransform(resize_rgb_image, resize_rgb_image_aspect_ratio, center_crop, normalization, transpose, to_array, to_tensor)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ResizeRGBImage"] = from_union([lambda x: to_class(ResizeRGBImage, x), from_none], self.resize_rgb_image)
        result["ResizeRGBImageAspectRatio"] = from_union([lambda x: to_class(ResizeRGBImageAspectRatio, x), from_none], self.resize_rgb_image_aspect_ratio)
        result["CenterCrop"] = from_union([lambda x: to_class(CenterCrop, x), from_none], self.center_crop)
        result["Normalization"] = from_union([lambda x: to_class(Normalization, x), from_none], self.normalization)
        result["Transpose"] = from_union([lambda x: to_class(Transpose, x), from_none], self.transpose)
        result["ToArray"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.to_array)
        result["ToTensor"] = from_union([lambda x: from_dict(lambda x: x, x), from_none], self.to_tensor)
        return result


@dataclass
class TransformationPipeline:
    steps: List[ImageTransform]

    @staticmethod
    def from_dict(obj: Any) -> 'TransformationPipeline':
        assert isinstance(obj, dict)
        steps = from_list(ImageTransform.from_dict, obj.get("steps"))
        return TransformationPipeline(steps)

    def to_dict(self) -> dict:
        result: dict = {}
        result["steps"] = from_list(lambda x: to_class(ImageTransform, x), self.steps)
        return result


@dataclass
class ModelConfig:
    channels: Channels
    image_size: ImageSize
    image_transformation: TransformationPipeline
    model_name: str
    model_url: str
    layer_name: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ModelConfig':
        assert isinstance(obj, dict)
        channels = Channels(obj.get("channels"))
        image_size = ImageSize.from_dict(obj.get("image_size"))
        image_transformation = TransformationPipeline.from_dict(obj.get("image_transformation"))
        model_name = from_str(obj.get("model_name"))
        model_url = from_str(obj.get("model_url"))
        layer_name = from_union([from_none, from_str], obj.get("layer_name"))
        return ModelConfig(channels, image_size, image_transformation, model_name, model_url, layer_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["channels"] = to_enum(Channels, self.channels)
        result["image_size"] = to_class(ImageSize, self.image_size)
        result["image_transformation"] = to_class(TransformationPipeline, self.image_transformation)
        result["model_name"] = from_str(self.model_name)
        result["model_url"] = from_str(self.model_url)
        result["layer_name"] = from_union([from_none, from_str], self.layer_name)
        return result


@dataclass
class GenericModelConfig:
    model_config: Optional[ModelConfig] = None
    model_architecture: Optional[ModelArchitecture] = None

    @staticmethod
    def from_dict(obj: Any) -> 'GenericModelConfig':
        assert isinstance(obj, dict)
        model_config = from_union([ModelConfig.from_dict, from_none], obj.get("ModelConfig"))
        model_architecture = from_union([ModelArchitecture, from_none], obj.get("ModelArchitecture"))
        return GenericModelConfig(model_config, model_architecture)

    def to_dict(self) -> dict:
        result: dict = {}
        result["ModelConfig"] = from_union([lambda x: to_class(ModelConfig, x), from_none], self.model_config)
        result["ModelArchitecture"] = from_union([lambda x: to_enum(ModelArchitecture, x), from_none], self.model_architecture)
        return result


@dataclass
class UpsertCollection:
    config: GenericModelConfig
    name: str

    @staticmethod
    def from_dict(obj: Any) -> 'UpsertCollection':
        assert isinstance(obj, dict)
        config = GenericModelConfig.from_dict(obj.get("config"))
        name = from_str(obj.get("name"))
        return UpsertCollection(config, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["config"] = to_class(GenericModelConfig, self.config)
        result["name"] = from_str(self.name)
        return result


def add_image_from_dict(s: Any) -> AddImage:
    return AddImage.from_dict(s)


def add_image_to_dict(x: AddImage) -> Any:
    return to_class(AddImage, x)


def common_from_dict(s: Any) -> Any:
    return s


def common_to_dict(x: Any) -> Any:
    return x


def remove_collection_from_dict(s: Any) -> RemoveCollection:
    return RemoveCollection.from_dict(s)


def remove_collection_to_dict(x: RemoveCollection) -> Any:
    return to_class(RemoveCollection, x)


def remove_image_from_dict(s: Any) -> RemoveImage:
    return RemoveImage.from_dict(s)


def remove_image_to_dict(x: RemoveImage) -> Any:
    return to_class(RemoveImage, x)


def search_image_from_dict(s: Any) -> SearchImage:
    return SearchImage.from_dict(s)


def search_image_to_dict(x: SearchImage) -> Any:
    return to_class(SearchImage, x)


def upsert_collection_from_dict(s: Any) -> UpsertCollection:
    return UpsertCollection.from_dict(s)


def upsert_collection_to_dict(x: UpsertCollection) -> Any:
    return to_class(UpsertCollection, x)
