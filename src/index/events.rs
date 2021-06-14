use crate::state::app::{CollectionName, GenericModelConfig};
use image::ImageFormat;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::serde_as;
use url::Url;

#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    AddImage(AddImage),
    RemoveImage(RemoveImage),
    SearchImage(SearchImage),
    UpsertCollection(UpsertCollection),
    RemoveCollection(RemoveCollection),
}

#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImageBytes {
    pub bytes: Vec<u8>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum ImageSource {
    ImageBytes(ImageBytes),
    Url(Url),
}

#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddImage {
    pub source: ImageSource,
    pub collection_name: CollectionName,
    pub id: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchImage {
    pub source: ImageSource,
    pub collection_name: CollectionName,
    pub n_results: usize,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemoveImage {
    pub index_name: String,
    pub id: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpsertCollection {
    pub name: String,
    pub config: GenericModelConfig,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemoveCollection {
    pub name: String,
}
