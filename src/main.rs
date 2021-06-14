use crate::image_transform::architectures::load_model_config;
use crate::image_transform::functions::read_rgb_image;
use crate::image_transform::models::{LoadedModel, ModelArchitecture};
use glob::glob;
use std::str::FromStr;
use std::time::Instant;
use tract_onnx::prelude::*;

mod image_transform;
mod index;
mod state;

fn main() -> Result<(), String> {
    let mut config = load_model_config(ModelArchitecture::ResNet152);
    // set layer to None to treat this as a normal prediction
    config.layer_name = None;
    let model = LoadedModel::new_from_config(config);

    let mut n_good = 0;
    let mut n_bad = 0;
    let start = Instant::now();

    for imagepath in glob("images/imagenet-sample-images/*.JPEG")
        .unwrap()
        .flatten()
    {
        let image = read_rgb_image(imagepath.to_str().unwrap());
        let image_tensor = model
            .config
            .image_transformation
            .transform_image(&image)
            .expect("Cannot transform image");

        let result = model.model.run(tvec!(image_tensor)).unwrap();

        // find and display the max value with its index
        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(2..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let predicted_class = best.unwrap().1 - 2;

        let class_id_str = imagepath
            .to_str()
            .unwrap()
            .split('/')
            .last()
            .unwrap()
            .split('.')
            .nth(0)
            .unwrap();
        let true_class = i32::from_str(class_id_str).unwrap();

        if true_class == predicted_class {
            n_good += 1;
        } else {
            n_bad += 1;
        }

        println!(
            "{:} true {:} predicted {:} acc {:} time per sample {:}",
            imagepath.to_str().unwrap(),
            true_class,
            predicted_class,
            (n_good as f32) / ((n_good + n_bad) as f32),
            start.elapsed().as_millis() as f32 / ((n_good + n_bad) as f32)
        );
    }

    Ok(())
}
