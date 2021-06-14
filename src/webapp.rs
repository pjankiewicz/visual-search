mod api;
mod image_transform;
mod index;
mod state;

use crate::image_transform::architectures::load_model_config;
use crate::image_transform::functions::read_rgb_image;
use crate::image_transform::models::{LoadedModel, ModelArchitecture};
use crate::index::events::{ImageSource, RemoveCollection, UpsertCollection};
use crate::state::app::GenericModelConfig;
use clap::App as ClapApp;
use glob::glob;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::Instant;
use tract_onnx::prelude::*;

use crate::index::events::{AddImage, RemoveImage, SearchImage};
use crate::state::app::EmbeddingApp;
use actix_web::web::Data;
use actix_web::{get, post, web, App, HttpRequest, HttpServer, Responder};
use std::fs::read_to_string;

#[derive(Serialize, Deserialize)]
pub struct ServerConfig {
    pub ip: String,
    pub port: u16,
}

#[derive(Serialize, Deserialize)]
pub struct AppConfig {
    pub n_workers: usize,
    pub server_config: ServerConfig,
}

#[post("/add_image")]
async fn add_image(state: web::Data<EmbeddingApp>, add_image: web::Json<AddImage>) -> String {
    state.add_image(add_image.into_inner());
    "ok".into()
}

#[post("/remove_image")]
async fn remove_image(
    state: web::Data<EmbeddingApp>,
    remove_image: web::Json<RemoveImage>,
) -> String {
    state.remove_image(remove_image.into_inner());
    "ok".into()
}

#[post("/search_image")]
async fn search_image(
    state: web::Data<EmbeddingApp>,
    search_image: web::Json<SearchImage>,
) -> String {
    state.search_image(search_image.into_inner());
    "ok".into()
}

#[post("/upsert_collection")]
async fn upsert_collection(
    state: web::Data<EmbeddingApp>,
    upsert_collection: web::Json<UpsertCollection>,
) -> String {
    state.upsert_collection(&upsert_collection.into_inner());
    "ok".into()
}

#[post("/remove_collection")]
async fn remove_collection(
    state: web::Data<EmbeddingApp>,
    remove_collection: web::Json<RemoveCollection>,
) -> String {
    state.remove_collection(&remove_collection.into_inner());
    "ok".into()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let matches = ClapApp::new("Visual Search ")
        .version("0.0.1")
        .author("LogicAI")
        .arg(
            clap::Arg::with_name("config")
                .long("config")
                .value_name("CONFIG")
                .help("Path to config file")
                .default_value("config.toml")
                .takes_value(true),
        )
        .get_matches();

    let app_config_str = read_to_string(
        matches
            .value_of("config")
            .expect("Argument config not specified")
            .to_string(),
    )
    .expect("Problems reading the file with configuration");

    let app_config: AppConfig = toml::from_str(&app_config_str)?;

    let full_address = format!(
        "{:}:{:}",
        &app_config.server_config.ip, &app_config.server_config.port
    );

    println!("Visual Search listening on {:}", full_address);
    let embedding_app = Data::new(EmbeddingApp::new(app_config.n_workers));
    HttpServer::new(move || {
        App::new()
            .app_data(embedding_app.clone())
            .service(add_image)
    })
    .bind(full_address)?
    .run()
    .await
}
