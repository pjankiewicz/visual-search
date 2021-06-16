# Visual Search in Rust

Rust web application for visual search. It is a component of [RecoAI](https://recoai.net) which is a fully featured engine
for e-commerce recommendation systems.

Visual Search in Rust is a single responsibility server/library to perform similar images queries.  It works by 
extracting features using selected deep learning model.

Features
-----------

- Ability to extract features from any ONNX model (https://github.com/onnx/models/tree/master/vision/classification)
- Image transformation pipeline written fully in Rust
- Supports indexing local image files (bytes) or remote (URL)
- Standalone server for image similarity search (using approximate nearest neighbors algorithm)
- Use as a server or as a library
- Multi-threaded and async indexing
- Python SDK

See example how to use the [SDK](sdk/sdk_example/visual_search_python_sdk_example.ipynb)



To do:
- [ ] persistance (right now the server is fully in-memory)