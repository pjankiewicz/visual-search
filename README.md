# image-embedding-rust

Code for the tutorial https://logicai.io/blog/extracting-image-embeddings/

Usage

```
./download_onnx_model.sh
cargo run -- --model-path mobilenetv2-7.onnx embed --normalize --image-size 224 --image-path cat.jpeg
```

It should print out the embedding for the image (1280 floats).

