mod image_transform;
mod index;
mod state;

use schemars::{schema_for, JsonSchema};

use crate::index::events::*;
use std::fs::File;
use std::io::Write;

fn generate_schema_for_event_type<T: JsonSchema>(name: &str) {
    let schema = schema_for!(T);
    let schema_str = serde_json::to_string_pretty(&schema).unwrap().to_string();
    let mut f =
        File::create(format!("sdk/schemas_raw/{:}.json", name)).expect("Could not create file");
    let _ = f.write_all(schema_str.as_bytes());
}

fn main() {
    generate_schema_for_event_type::<AddImage>("add_image");
    generate_schema_for_event_type::<RemoveImage>("remove_image");
    generate_schema_for_event_type::<UpsertCollection>("upsert_collection");
    generate_schema_for_event_type::<RemoveCollection>("remove_collection");
    generate_schema_for_event_type::<SearchImage>("search_image");
}
