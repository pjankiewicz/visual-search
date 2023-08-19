use bytes::Bytes;
use image::RgbImage;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn remove_non_alphanum(s: &str) -> String {
    let s_new: String = s
        .chars()
        .map(|x| match x {
            '0'..='9' => x,
            'A'..='Z' => x.to_ascii_lowercase(),
            'a'..='z' => x,
            ' ' => '_',
            _ => ' ',
        })
        .collect();
    s_new.replace(" ", "_")
}

pub fn model_filename(name: &str, extension: &str) -> String {
    let clean_name = remove_non_alphanum(name);
    format!("models/{}.{}", clean_name, extension)
}

pub fn save_file_get(url: &str, path: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .referer(false)
        .build()
        .map_err(|e| e.to_string())?;

    let response = client.get(url).send().map_err(|e| e.to_string())?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().unwrap();
        return Err(if text.is_empty() {
            status.to_string()
        } else {
            text
        });
    }

    if !Path::new("models/").exists() {
        fs::create_dir("models").map_err(|e| e.to_string())?;
    }
    let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
    out.write_all(&response.bytes().expect("Failed to convert to bytes"));

    Ok(())
}

pub fn read_bytes_url(url: &str) -> reqwest::Result<Bytes> {
    let client = reqwest::blocking::Client::builder()
        .referer(false)
        .build()
        .map_err(|e| e)?;
    let response = client.get(url).send().map_err(|e| e)?;
    response.bytes()
}

pub fn image_from_bytes(bytes: &Bytes) -> Result<RgbImage, Box<dyn Error>> {
    let dynimg = image::load_from_memory(bytes).unwrap();
    let img = dynimg.to_rgb8();
    Ok(img)
}
