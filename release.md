# Release

```shell
cargo build --release --target wasm32-unknown-unknown

wasm-bindgen --no-typescript --out-name bevy_game --out-dir wasm --target web target/wasm32-unknown-unknown/release/beamed.wasm

cp -r assets wasm/
```

## Local testing

```shell
cd wasm/

basic-http-server -x
```
