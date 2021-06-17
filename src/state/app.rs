use crate::image_transform::models::{LoadedModel, ModelArchitecture, ModelConfig};
use crate::image_transform::utils::{image_from_bytes, read_bytes_url};
use crate::index::db::VectorIndex;
use crate::index::events::{
    AddImage, ImageBytes, ImageSource, RemoveCollection, RemoveImage, SearchImage, UpsertCollection,
};
use crate::state::work_queue::WorkQueue;
use image::{ImageBuffer, Rgb};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
use std::thread;
use std::thread::JoinHandle;

pub type CollectionName = String;
pub type ImageId = String;
pub type ModelId = String;

#[derive(Clone, PartialEq)]
pub enum Job {
    AddImage(AddImage),
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct SingleImageResult {
    pub id: String,
    pub similarity: u32,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImageResult {
    pub collection_name: String,
    pub results: Vec<SingleImageResult>,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub enum GenericModelConfig {
    ModelConfig(ModelConfig),
    ModelArchitecture(ModelArchitecture),
}

#[derive(Clone)]
pub struct Collection {
    pub name: String,
    pub model_config: GenericModelConfig,
    pub model: LoadedModel,
    pub index: VectorIndex,
}

impl Collection {
    pub fn new(name: &str, model_config: &GenericModelConfig) -> Self {
        let model = match model_config {
            GenericModelConfig::ModelConfig(config) => {
                LoadedModel::new_from_config((*config).clone())
            }
            GenericModelConfig::ModelArchitecture(architecture) => {
                LoadedModel::new_from_architecture((*architecture).clone())
            }
        };

        Collection {
            name: name.to_string(),
            model_config: model_config.clone(),
            model: model,
            index: VectorIndex::new(),
        }
    }
}

pub struct EmbeddingApp {
    pub n_workers: usize,
    pub job_queue: WorkQueue<Job>,
    pub images: Arc<RwLock<HashMap<ImageId, ImageBytes>>>,
    pub collections: Arc<RwLock<HashMap<CollectionName, Collection>>>,
}

impl EmbeddingApp {
    pub fn new(n_workers: usize) -> Self {
        Self {
            n_workers,
            job_queue: WorkQueue::new(),
            images: Arc::new(Default::default()),
            collections: Arc::new(Default::default()),
        }
    }

    pub fn upsert_collection(&self, upsert_collection: &UpsertCollection) {
        let mut collections = self.collections.write().unwrap();
        if collections.contains_key(&upsert_collection.name) {
            // create a task to rebuild a collection
            // I think for now we can disable this
        } else {
            let collection = Collection::new(&upsert_collection.name, &upsert_collection.config);
            collections.insert(upsert_collection.name.clone(), collection);
        }
    }

    pub fn remove_collection(&self, remove_collection: &RemoveCollection) {
        let mut collections = self.collections.write().unwrap();
        collections.remove(&remove_collection.name.clone());
    }

    pub fn add_image(&self, add_image: AddImage) -> Result<(), Box<dyn Error>> {
        self.job_queue.add_work(Job::AddImage(add_image));
        Ok(())
    }

    pub fn remove_image(&self, remove_image: RemoveImage) -> Result<(), Box<dyn Error>> {
        let mut collections = self.collections.write().map_err(|_| "RwLock Error")?;
        collections
            .entry(remove_image.collection_name.clone())
            .and_modify(|c| c.index.remove(remove_image.id));
        Ok(())
    }

    pub fn search_image(&self, search_image: SearchImage) -> Result<ImageResult, Box<dyn Error>> {
        let collections = self.collections.write().map_err(|_| "RwLock Error")?;
        if let Some(collection) = collections.get(&search_image.collection_name) {
            let image = EmbeddingApp::image_source_to_rgb_image(&search_image.source)?;
            println!("Extracting features");
            let features = collection.model.extract_features(image)?;
            println!("Features len {}", features.len());
            let results: Vec<_> = collection
                .index
                .search(&features)
                .iter()
                .map(|result| SingleImageResult {
                    id: result.id.clone(),
                    similarity: result.distance,
                })
                .collect();
            Ok(ImageResult {
                collection_name: search_image.collection_name.clone(),
                results,
            })
        } else {
            Ok(ImageResult {
                collection_name: search_image.collection_name.clone(),
                results: vec![],
            })
        }
    }

    pub fn start_workers(&self) -> Vec<JoinHandle<()>> {
        println!("Starting workers");
        let mut handles: Vec<_> = Vec::new();
        for n in 0..self.n_workers {
            println!("Starting worker {}", n);
            let tq = self.job_queue.clone();
            let collections = self.collections.clone();
            let handle = thread::spawn(move || loop {
                if let Some(job) = tq.get_work() {
                    println!("Queue length {}", tq.inner.lock().unwrap().len());
                    match job {
                        Job::AddImage(add_image) => {
                            EmbeddingApp::add_image_to_collection(collections.clone(), &add_image);
                            ()
                        }
                    }
                }
                std::thread::yield_now();
                std::thread::sleep(std::time::Duration::from_millis(10));
            });
            handles.push(handle);
        }
        handles
    }

    fn add_image_to_collection(
        collections: Arc<RwLock<HashMap<String, Collection>>>,
        add_image: &AddImage,
    ) -> Result<(), Box<dyn Error>> {
        let image = EmbeddingApp::image_source_to_rgb_image(&add_image.source)?;

        println!("Locking collections for reading");
        let collection_read = collections.read().map_err(|_| "RwLock Error")?;

        println!("Extracting features");
        let features = if let Some(collection) = collection_read.get(&add_image.collection_name) {
            let features = collection.model.extract_features(image)?;
            Some(features)
        } else {
            None
        };
        drop(collection_read);

        println!("Writing features to the index");
        if let Some(features) = features {
            let collection_write = collections.write().map_err(|_| "RwLock Error")?;
            if let Some(collection) = collection_write.get(&add_image.collection_name) {
                collection.index.insert(features, add_image.id.clone());
            }
            drop(collection_write);
        };
        println!("Finished");

        Ok(())
    }

    fn image_source_to_rgb_image(
        image_source: &ImageSource,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        println!("Add Image");
        let image_bytes = match &image_source {
            ImageSource::ImageBytes(image_bytes) => image_bytes.bytes.clone(),
            ImageSource::Url(url) => read_bytes_url(url.as_str())?.to_vec(),
        };
        println!("Converting bytes to RgbImage");
        let bytes = bytes::Bytes::from(image_bytes);
        let image = image_from_bytes(&bytes)?;
        println!("Converted");
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Url;
    use std::str::FromStr;

    #[test]
    fn test_app() {
        let app = EmbeddingApp::new(4);
        app.start_workers();

        app.upsert_collection(&UpsertCollection {
            name: "images".to_string(),
            config: GenericModelConfig::ModelArchitecture(ModelArchitecture::MobileNetV2),
        });

        app.add_image(AddImage{
            source: ImageSource::Url(Url::from_str("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG").unwrap()),
            collection_name: "images".into(),
            id: "goldfish".into()
        });

        app.add_image(AddImage{
            source: ImageSource::Url(Url::from_str("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01491361_tiger_shark.JPEG").unwrap()),
            collection_name: "images".into(),
            id: "shark".into()
        });

        app.add_image(AddImage{
            source: ImageSource::Url(Url::from_str("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01496331_electric_ray.JPEG").unwrap()),
            collection_name: "images".into(),
            id: "ray".into()
        });

        app.add_image(AddImage{
            source: ImageSource::Url(Url::from_str("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01622779_great_grey_owl.JPEG").unwrap()),
            collection_name: "images".into(),
            id: "owl".into()
        });
    }
}
