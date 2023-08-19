use crate::image_transform::architectures::load_model_config;
use crate::image_transform::pipeline::{ImageSize, TransformationPipeline};
use crate::image_transform::utils::{model_filename, save_file_get};
use image::RgbImage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tract_nnef::prelude::*;
use tract_onnx::prelude::*;

pub type TractSimplePlan =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub enum Channels {
    CWH,
    WHC,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub enum ModelType {
    ONNX,
    NNEF,
}

impl ModelType {
    pub fn to_extension(&self) -> String {
        match self {
            ModelType::ONNX => "onnx".to_string(),
            ModelType::NNEF => "nnef.tgz".to_string(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_url: String,
    pub model_type: ModelType,
    pub image_transformation: TransformationPipeline,
    pub image_size: ImageSize,
    pub layer_name: Option<String>,
    pub channels: Channels,
}

#[derive(Clone)]
pub struct LoadedModel {
    pub config: ModelConfig,
    pub model: TractSimplePlan,
}

impl LoadedModel {
    pub fn new_from_architecture(architecture: ModelArchitecture) -> Self {
        let config = load_model_config(architecture);
        let model = LoadedModel::load_model(&config);
        Self { config, model }
    }

    pub fn new_from_config(config: ModelConfig) -> Self {
        let model = LoadedModel::load_model(&config);
        Self { config, model }
    }

    pub fn load_model(config: &ModelConfig) -> TractSimplePlan {
        let name = config.model_name.clone();
        let url = config.model_url.clone();
        let extension = config.model_type.to_extension();
        let filename = model_filename(&name, &extension);
        if !Path::new(&filename).exists() {
            println!("Downloading model file");
            save_file_get(&url, &filename);
        } else {
            println!("Skipping download");
        }

        let input_shape = match config.channels {
            Channels::CWH => tvec!(1, 3, config.image_size.width, config.image_size.height),
            Channels::WHC => tvec!(1, config.image_size.width, config.image_size.height, 3),
        };

        // let mut model = match config.model_type {
        //     ModelType::NNEF => {
        //         tract_nnef::nnef()
        //             .model_for_path(&filename)
        //             .expect("Cannot read model")
        //             .with_input_fact(0, TypedFact::dt_shape(f32::datum_type(), input_shape))
        //             .unwrap()
        //     },
        //     ModelType::ONNX => {
        //         let a = tract_onnx::onnx()
        //             .model_for_path(&filename)
        //             .expect("Cannot read model")
        //             .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
        //             .unwrap()
        //     }
        // };

        let mut model = tract_onnx::onnx()
            .model_for_path(&filename)
            .expect("Cannot read model")
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .unwrap();

        if let Some(layer_name) = config.layer_name.clone() {
            let node_names: Vec<&str> = model.node_names().collect::<Vec<&str>>().clone();
            println!("Available nodes {:?}", node_names);
            model = model.with_output_names(vec![layer_name]).unwrap()
        }

        model.into_optimized().unwrap().into_runnable().unwrap()
    }

    pub fn extract_features(&self, image: RgbImage) -> Result<Vec<f64>, String> {
        println!("Transforming the image");
        let image_tensor = self
            .config
            .image_transformation
            .transform_image(&image)
            .expect("Cannot transform image");
        println!("Running the model");
        let result = self
            .model
            .run(tvec!(image_tensor.into()))
            .expect("Cannot run model");
        let features: Vec<f64> = result[0]
            .to_array_view::<f32>()
            .expect("Cannot extract feature vector")
            .iter()
            .cloned()
            .map(|v| v as f64)
            .collect();
        Ok(features)
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub enum ModelArchitecture {
    SqueezeNet,
    MobileNetV2,
    ResNet152,
    EfficientNetLite4,
    GoogleNet,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_transform::functions::read_rgb_image;

    #[test]
    fn test_feature_extraction() {
        let model = LoadedModel::new_from_architecture(ModelArchitecture::EfficientNetLite4);
        let image = read_rgb_image("images/cat.jpeg");
        let features = model.extract_features(image).unwrap();
        assert_eq!(features.len(), 1280);
    }
}
