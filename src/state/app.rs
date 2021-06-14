use crate::image_transform::models::{
    LoadedModel, ModelArchitecture, ModelConfig, TractSimplePlan,
};
use crate::image_transform::utils::{image_from_bytes, read_bytes_url};
use crate::index::db::VectorIndex;
use crate::index::events::{AddImage, ImageBytes, ImageSource, UpsertCollection};
use crate::state::work_queue::WorkQueue;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::thread;

pub type CollectionName = String;
pub type ImageId = String;
pub type ModelId = String;

#[derive(Clone, PartialEq)]
pub enum Job {
    AddImage(AddImage),
}

#[derive(Clone)]
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

pub struct AppState {
    pub n_workers: usize,
    pub job_queue: WorkQueue<Job>,
    pub images: Arc<RwLock<HashMap<ImageId, ImageBytes>>>,
    pub collections: Arc<RwLock<HashMap<CollectionName, Collection>>>,
}

impl AppState {
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
            let model = match &upsert_collection.config {
                GenericModelConfig::ModelConfig(config) => {
                    LoadedModel::new_from_config((*config).clone())
                }
                GenericModelConfig::ModelArchitecture(architecture) => {
                    LoadedModel::new_from_architecture((*architecture).clone())
                }
            };

            let collection = Collection {
                name: upsert_collection.name.clone(),
                model_config: upsert_collection.config.clone(),
                model: model,
                index: VectorIndex::new(),
            };
            collections.insert(upsert_collection.name.clone(), collection);
        }
    }

    pub fn add_image(&self, add_image: AddImage) {
        self.job_queue.add_work(Job::AddImage(add_image));
    }

    pub fn rebuild_existing_collection(
        &self,
        name: CollectionName,
        generic_model_config: GenericModelConfig,
    ) {
        //
    }

    pub fn start_workers(&self) {
        for _ in 0..self.n_workers {
            let tq = self.job_queue.clone();
            let collections = self.collections.clone();
            // let handle = thread::spawn(move || {
            //     loop {
            //         if let Some(job) = tq.get_work() {
            //             match job {
            //                 Job::AddImage(add_image) => {
            //                     let collections_read = self.collections.read().unwrap();
            //                     if let Some(collection) = collections_read.get(&add_image.collection_name) {
            //                         let image_bytes = match add_image.source {
            //                             ImageSource::ImageBytes(image_bytes) => image_bytes.bytes,
            //                             ImageSource::Url(url) => read_bytes_url(url.as_str()).unwrap().to_vec()
            //                         };
            //                         let image = image_from_bytes(bytes::Bytes::from(image_bytes)).unwrap();
            //                         let features = collection.model.extract_features(image);
            //                     }
            //                 },
            //                 _ => {}
            //             }
            //         }
            //         std::thread::yield_now();
            //         std::thread::sleep(std::time::Duration::from_millis(100));
            //     }
            // });
        }
    }
}
