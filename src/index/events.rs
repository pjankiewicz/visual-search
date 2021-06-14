use crate::state::app::{CollectionName, GenericModelConfig};
use image::ImageFormat;
use url::Url;

#[derive(Clone)]
pub enum Event {
    AddImage(AddImage),
    RemoveImage(DeleteImage),
    UpsertIndex(UpsertCollection),
    DeleteIndex(DeleteImage),
}

#[derive(Clone, PartialEq)]
pub struct ImageBytes {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
}

#[derive(Clone, PartialEq)]
pub enum ImageSource {
    ImageBytes(ImageBytes),
    Url(Url),
}

#[derive(Clone, PartialEq)]
pub struct AddImage {
    pub source: ImageSource,
    pub collection_name: CollectionName,
    pub id: String,
}

#[derive(Clone)]
pub struct DeleteImage {
    pub index_name: String,
    pub id: String,
}

#[derive(Clone)]
pub struct UpsertCollection {
    pub name: String,
    pub config: GenericModelConfig,
}

#[derive(Clone)]
pub struct DeleteIndex {
    pub name: String,
}
