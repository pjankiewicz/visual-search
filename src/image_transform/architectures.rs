use crate::image_transform::models::{Channels, ModelArchitecture, ModelConfig};
use crate::image_transform::pipeline::{
    CenterCrop, ImageSize, Normalization, ResizeRGBImage, ResizeRGBImageAspectRatio, ToArray,
    ToTensor, TransformationPipeline, Transpose,
};
use image::imageops::FilterType;

pub fn load_model_config(model: ModelArchitecture) -> ModelConfig {
    match model {
        // Top-1 accuracy 1000 imagenet: 61.9% (39ms per image)
        ModelArchitecture::SqueezeNet => ModelConfig {
            model_name: "SqueezeNet".into(),
            model_url: "https://github.com/onnx/models/blob/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx?raw=true".into(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    ResizeRGBImageAspectRatio { image_size: ImageSize { width: 224, height: 224 }, scale: 87.5, filter: FilterType::Nearest }.into(),
                    CenterCrop { crop_size: ImageSize {width: 224, height: 224} }.into(),
                    ToArray {}.into(),
                    Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }.into(),
                    ToTensor {}.into(),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("squeezenet0_pool3_fwd".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 79.8% (75ms per image)
        ModelArchitecture::MobileNetV2 => ModelConfig {
            model_name: "MobileNetV2".into(),
            model_url: "https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx?raw=true".into(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    ResizeRGBImage { image_size: ImageSize { width: 224, height: 224 }, filter: FilterType::Nearest }.into(),
                    ToArray {}.into(),
                    Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }.into(),
                    ToTensor {}.into(),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("Reshape_103".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 90.9% (477ms per image)
        ModelArchitecture::ResNet152 => ModelConfig {
            model_name: "ResNet152".to_string(),
            model_url: "https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet152-v2-7.onnx?raw=true".to_string(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    ResizeRGBImageAspectRatio { image_size: ImageSize { width: 224, height: 224 }, scale: 87.5, filter: FilterType::Nearest }.into(),
                    CenterCrop { crop_size: ImageSize {width: 224, height: 224} }.into(),
                    ToArray {}.into(),
                    Normalization { sub: [0.485, 0.456, 0.406], div: [0.229, 0.224, 0.225], zeroone: true }.into(),
                    ToTensor {}.into(),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("resnetv27_flatten0_reshape0".to_string()),
            channels: Channels::CWH
        },
        // Top-1 accuracy 1000 imagenet: 89.6% (230ms per image)
        ModelArchitecture::EfficientNetLite4 => ModelConfig {
            model_name: "EfficientNet-Lite4".to_string(),
            model_url: "https://github.com/onnx/models/blob/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx?raw=true".to_string(),
            image_transformation: TransformationPipeline {
                steps: vec![
                    ResizeRGBImageAspectRatio { image_size: ImageSize { width: 224, height: 224 }, scale: 87.5, filter: FilterType::Nearest }.into(),
                    CenterCrop { crop_size: ImageSize {width: 224, height: 224} }.into(),
                    ToArray {}.into(),
                    Normalization { sub: [127.0, 127.0, 127.0], div: [128.0, 128.0, 128.0], zeroone: false }.into(),
                    ToTensor {}.into(),
                    Transpose { axes: [0, 2, 3, 1] }.into(),
                ]
            },
            image_size: ImageSize { width: 224, height: 224 },
            layer_name: Some("efficientnet-lite4/model/head/Squeeze".into()),
            channels: Channels::WHC
        }
    }
}
