use crate::state::app::{CollectionName, GenericModelConfig};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Clone, Serialize, Deserialize)]
pub enum Event {
    AddImage(AddImage),
    RemoveImage(RemoveImage),
    SearchImage(SearchImage),
    UpsertCollection(UpsertCollection),
    RemoveCollection(RemoveCollection),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ImageBytes {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum ImageSource {
    ImageBytes(ImageBytes),
    Url(Url),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddImage {
    pub source: ImageSource,
    pub collection_name: CollectionName,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchImage {
    pub source: ImageSource,
    pub collection_name: CollectionName,
    pub n_results: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RemoveImage {
    pub collection_name: String,
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
