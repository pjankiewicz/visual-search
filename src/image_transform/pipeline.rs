use crate::image_transform::pipeline::tract_ndarray::Ix4;
use enum_dispatch::enum_dispatch;
use image::imageops::{crop, resize, FilterType};
use image::RgbImage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use tract_onnx::prelude::tract_ndarray::Array4;
use tract_onnx::prelude::{tract_ndarray, Tensor};
use tract_onnx::tract_core::ndarray::Array;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub enum ImageTransformResult {
    RgbImage(RgbImage),
    Array4(Array4<f32>),
    Tensor(Tensor),
}

#[enum_dispatch]
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub enum ImageTransform {
    ResizeRGBImage(ResizeRGBImage),
    ResizeRGBImageAspectRatio(ResizeRGBImageAspectRatio),
    CenterCrop(CenterCrop),
    Normalization(Normalization),
    Transpose(Transpose),
    ToArray(ToArray),
    ToTensor(ToTensor),
}

impl ImageTransformResult {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            ImageTransformResult::RgbImage(image) => {
                let (width, height) = image.dimensions();
                vec![width as usize, height as usize]
            }
            ImageTransformResult::Array4(array) => {
                let shape: Vec<usize> = array.shape().iter().map(|v| v.clone() as usize).collect();
                shape
            }
            ImageTransformResult::Tensor(tensor) => {
                let shape: Vec<usize> = tensor.shape().iter().map(|v| v.clone() as usize).collect();
                shape
            }
        }
    }
}

impl From<RgbImage> for ImageTransformResult {
    fn from(rgb_image: RgbImage) -> Self {
        ImageTransformResult::RgbImage(rgb_image)
    }
}

impl From<Tensor> for ImageTransformResult {
    fn from(tensor: Tensor) -> Self {
        ImageTransformResult::Tensor(tensor)
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct TransformationPipeline {
    pub steps: Vec<ImageTransform>,
}

impl TransformationPipeline {
    pub fn transform_image(&self, image: &RgbImage) -> Result<Tensor, &'static str> {
        let mut result = ImageTransformResult::RgbImage(image.clone());

        for step in &self.steps {
            result = step.transform(result)?;
            // println!("Shape {:?}", result.shape());
        }

        let to_tensor = ToTensor {};
        result = to_tensor.transform(result)?;

        match result {
            ImageTransformResult::Tensor(t) => Ok(t),
            _ => Err("Should be converted to tensor already"),
        }
    }
}

#[enum_dispatch(ImageTransform)]
pub trait GenericTransform {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str>;
}

#[serde_as]
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResizeRGBImage {
    pub image_size: ImageSize,
    #[serde(with = "FilterOption")]
    pub filter: FilterType,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(remote = "FilterType")]
pub enum FilterOption {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

impl GenericTransform for ResizeRGBImage {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => Ok(resize(
                &image,
                self.image_size.width as u32,
                self.image_size.width as u32,
                FilterType::Triangle,
            )
            .into()),
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

// Resizes the image to a size but keeps the aspect ratio
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResizeRGBImageAspectRatio {
    pub image_size: ImageSize,
    pub scale: f32,
    #[serde(with = "FilterOption")]
    pub filter: FilterType,
}

impl GenericTransform for ResizeRGBImageAspectRatio {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let (height, width) = image.dimensions();
                let height = height as f32;
                let width = width as f32;
                let new_height = 100.0 * (self.image_size.height as f32) / self.scale;
                let new_width = 100.0 * (self.image_size.width as f32) / self.scale;

                let (final_height, final_width) = if height > width {
                    (new_width, new_height * height / width)
                } else {
                    (new_width * width / height, new_width)
                };

                Ok(resize(&image, final_width as u32, final_height as u32, self.filter).into())
            }
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

// Resizes the image to a size but keeps the aspect ratio
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct CenterCrop {
    pub crop_size: ImageSize,
}

impl GenericTransform for CenterCrop {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let (height, width) = image.dimensions();
                let left = (width - self.crop_size.width as u32) / 2;
                let top = (height - self.crop_size.height as u32) / 2;
                let mut image_cropped = image.clone();
                let image_cropped_new = crop(
                    &mut image_cropped,
                    top as u32,
                    left as u32,
                    self.crop_size.width as u32,
                    self.crop_size.height as u32,
                );
                Ok(image_cropped_new.to_image().into())
            }
            ImageTransformResult::Tensor(_) => Err("Image resize not implemented for Tensor"),
            ImageTransformResult::Array4(_) => Err("Image resize not implemented for Array4"),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct Normalization {
    pub sub: [f32; 3],
    pub div: [f32; 3],
    pub zeroone: bool,
}

impl GenericTransform for Normalization {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(_) => Err("Not implemented"),
            ImageTransformResult::Tensor(_) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let sub = Array::from_shape_vec((1, 3, 1, 1), self.sub.to_vec())
                    .expect("Wrong conversion to array");
                let div = Array::from_shape_vec((1, 3, 1, 1), self.div.to_vec())
                    .expect("Wrong conversion to array");
                let new_arr = if self.zeroone {
                    (arr / 255.0 - sub) / div
                } else {
                    (arr - sub) / div
                };
                Ok(ImageTransformResult::Array4(new_arr))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct Transpose {
    pub axes: [usize; 4],
}

impl GenericTransform for Transpose {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(_) => Err("Not implemented"),
            ImageTransformResult::Array4(arr) => {
                let arr = arr.permuted_axes(self.axes);
                Ok(ImageTransformResult::Array4(arr))
            }
            ImageTransformResult::Tensor(tensor) => {
                // note that the same operation on Tensor is not safe as it is on Array4
                let tensor = tensor
                    .permute_axes(&self.axes)
                    .expect("Transpose should match the shape of the tensor");
                Ok(ImageTransformResult::Tensor(tensor))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToTensor {}

impl GenericTransform for ToTensor {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let shape = image.dimensions();
                let tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
                    (1 as usize, 3 as usize, shape.0 as usize, shape.1 as usize),
                    |(_, c, y, x)| image[(x as _, y as _)][c] as f32,
                )
                .into();
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Tensor(tensor) => {
                // already a tensor
                Ok(ImageTransformResult::Tensor(tensor))
            }
            ImageTransformResult::Array4(arr4) => Ok(ImageTransformResult::Tensor(arr4.into())),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToArray {}

impl GenericTransform for ToArray {
    fn transform(&self, input: ImageTransformResult) -> Result<ImageTransformResult, &'static str> {
        match input {
            ImageTransformResult::RgbImage(image) => {
                let shape = image.dimensions();
                let arr = tract_ndarray::Array4::from_shape_fn(
                    (1 as usize, 3 as usize, shape.0 as usize, shape.1 as usize),
                    |(_, c, y, x)| image[(x as _, y as _)][c] as f32,
                );
                Ok(ImageTransformResult::Array4(arr))
            }
            ImageTransformResult::Tensor(tensor) => {
                let dyn_arr = tensor
                    .into_array::<f32>()
                    .expect("Cannot convert tensor to Array4");
                let arr4 = dyn_arr
                    .into_dimensionality::<Ix4>()
                    .expect("Cannot convert dynamic Array to Array4");
                Ok(ImageTransformResult::Array4(arr4))
            }
            ImageTransformResult::Array4(arr4) => {
                // already an array
                Ok(ImageTransformResult::Tensor(arr4.into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use image::imageops::FilterType;

    use crate::image_transform::pipeline::{ImageSize, ResizeRGBImage, TransformationPipeline};

    use super::*;
    use crate::image_transform::functions::read_rgb_image;
    use tract_onnx::prelude::*;

    #[test]
    fn test_resize() {
        let pipeline = TransformationPipeline {
            steps: vec![
                ResizeRGBImage {
                    image_size: ImageSize {
                        width: 224,
                        height: 224,
                    },
                    filter: FilterType::Nearest,
                }
                .into(),
                ToTensor {}.into(),
            ],
        };
        let image = read_rgb_image("images/cat.jpeg");
        let result = pipeline
            .transform_image(&image)
            .expect("Cannot transform image");
        assert_eq!(result.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_resize_permute() {
        let pipeline = TransformationPipeline {
            steps: vec![
                ResizeRGBImage {
                    image_size: ImageSize {
                        width: 224,
                        height: 224,
                    },
                    filter: FilterType::Nearest,
                }
                .into(),
                ToTensor {}.into(),
                Transpose { axes: [0, 2, 3, 1] }.into(),
            ],
        };
        let image = read_rgb_image("images/cat.jpeg");
        let result = pipeline
            .transform_image(&image)
            .expect("Cannot transform image");
        assert_eq!(result.shape(), &[1, 224, 224, 3]);
    }

    #[test]
    fn test_classification() {
        let pipeline = TransformationPipeline {
            steps: vec![
                ResizeRGBImage {
                    image_size: ImageSize {
                        width: 224,
                        height: 224,
                    },
                    filter: FilterType::Nearest,
                }
                .into(),
                ToArray {}.into(),
                Normalization {
                    sub: [0.485, 0.456, 0.406],
                    div: [0.229, 0.224, 0.225],
                    zeroone: true,
                }
                .into(),
                ToTensor {}.into(),
            ],
        };
        let image = read_rgb_image("images/cat.jpeg");
        let image_tensor = pipeline
            .transform_image(&image)
            .expect("Cannot transform image");

        let model = tract_onnx::onnx()
            .model_for_path("models/mobilenetv2-7.onnx".to_string())
            .expect("Cannot read model")
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)),
            )
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();

        let result = model.run(tvec!(image_tensor)).unwrap();

        // find and display the max value with its index
        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(2..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // this is classified as a lynx which is close enough I guess
        assert_eq!(best.unwrap().1, 287);
    }
}
