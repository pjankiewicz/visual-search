use image::RgbImage;

pub fn read_rgb_image(image_path: &str) -> RgbImage {
    image::open(image_path).unwrap().to_rgb8()
}
