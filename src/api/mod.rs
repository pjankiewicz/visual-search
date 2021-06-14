pub mod http_server;

use crate::index::events::{AddImage, RemoveImage, SearchImage};
use crate::state::app::EmbeddingApp;
use actix_web::web::Data;
use actix_web::{get, post, web, App, HttpRequest, HttpServer, Responder};

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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let embedding_app = Data::new(EmbeddingApp::new(4));
    HttpServer::new(move || {
        App::new()
            .app_data(embedding_app.clone())
            .service(add_image)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
