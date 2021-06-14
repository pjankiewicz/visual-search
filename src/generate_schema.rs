mod api;
mod image_transform;
mod index;
mod state;

use schemars::schema_for;

use crate::index::events::Event;
use std::fs::File;
use std::io::Write;

fn generate_schema_for_event_type(event_type: Event) {
    let schema = match event_type {
        Event::CategoryArticlePageView => schema_for!(CategoryArticlePageView),
        Event::DetailsArticleView => schema_for!(DetailArticleView),
        Event::OtherArticleInteraction => schema_for!(OtherArticleInteraction),
    };
    let schema_str = serde_json::to_string_pretty(&schema).unwrap().to_string();
    let mut f = File::create(format!("sdk/schemas/{:}.json", event_type.to_string()))
        .expect("Could not create file");
    let _ = f.write_all(schema_str.as_bytes());
}

fn main() {
    generate_schema_for_event_type(EventType::AddToCart);
    generate_schema_for_event_type(EventType::AddToList);
    generate_schema_for_event_type(EventType::CartPageView);
    generate_schema_for_event_type(EventType::CategoryPageView);
    generate_schema_for_event_type(EventType::CheckoutStart);
    generate_schema_for_event_type(EventType::DetailProductView);
    generate_schema_for_event_type(EventType::HomePageView);
    generate_schema_for_event_type(EventType::ImageInteraction);
    generate_schema_for_event_type(EventType::ListView);
    generate_schema_for_event_type(EventType::OtherInteraction);
    generate_schema_for_event_type(EventType::PageVisit);
    generate_schema_for_event_type(EventType::PurchaseComplete);
    generate_schema_for_event_type(EventType::RateProduct);
    generate_schema_for_event_type(EventType::RecoRequest);
    generate_schema_for_event_type(EventType::RecoShow);
    generate_schema_for_event_type(EventType::SmartSearchRequest);
    generate_schema_for_event_type(EventType::SmartSearchShow);
    generate_schema_for_event_type(EventType::RemoveFromCart);
    generate_schema_for_event_type(EventType::RemoveFromList);
    generate_schema_for_event_type(EventType::SearchItems);
    generate_schema_for_event_type(EventType::SortItems);
    generate_schema_for_event_type(EventType::DetailsArticleView);
    generate_schema_for_event_type(EventType::CategoryArticlePageView);
    generate_schema_for_event_type(EventType::OtherArticleInteraction);
    generate_schema_for_event_type(EventType::ItemUpsert);
    generate_schema_for_event_type(EventType::ItemRemove);
    generate_schema_for_event_type(EventType::ArticleUpsert);
    generate_schema_for_event_type(EventType::ArticleRemove);
    generate_schema_for_event_type(EventType::PlacementUpsert);
    generate_schema_for_event_type(EventType::PlacementRemove);
    generate_schema_for_event_type(EventType::ChangeItemStockState);
    generate_schema_for_event_type(EventType::OfflineRecommendationsUpsert);
    generate_schema_for_event_type(EventType::OfflineRecommendationsRemove);
    generate_schema_for_event_type(EventType::RankingModelTrainRequest);
    generate_schema_for_event_type(EventType::UnknownEvent);
    generate_other_schemas();
    generate_api_settings();
}
