# AlphaBlokus browser frontend

This is a static browser application: it does not use the Rust `web-play`
server or make `/api` calls. The ONNX model pair and canonical move table are
checked into `static` with Git LFS and are served alongside the app.

Inference defaults to WebGPU when it is available, except on iOS, where it
defaults to CPU WASM. Override the selection with `?backend=webgpu` or
`?backend=wasm`. Add `debug=true` to display the active backend, for example
`?backend=wasm&debug=true`. The selected runtime is loaded dynamically; the
unused ONNX Runtime WASM binary is not fetched or initialized.

```bash
cd frontend
npm install
npm run build:wasm
npm run dev
```

The WASM build uses the Rustup toolchain explicitly, so it continues to work
when a Homebrew Rust installation appears earlier on `PATH`. Install the
target once if needed: `rustup target add wasm32-unknown-unknown`.

The browser and native application both load `static/move_data/full.bin`.
Regenerate it only when the precomputed move data changes. GitHub Pages copies
the checked-in artifact; it never regenerates it:

```bash
cargo run --release --bin generate-move-data -- \
  --config configs/generate_move_data/full.toml
```

## PostHog

Copy `.env.example` to `.env.local` and add the project token and ingestion
host from the PostHog project settings:

```sh
VITE_POSTHOG_PROJECT_TOKEN=phc_...
VITE_POSTHOG_HOST=https://us.i.posthog.com
```

The SDK and React provider are initialized only when
`VITE_POSTHOG_PROJECT_TOKEN` is set. Without a token, PostHog is disabled and
the frontend does not send analytics requests. The GitHub Pages workflow reads
both values from repository Actions secrets with the same names.

When configured, PostHog captures product analytics, page views and exits,
session replays, console output, automatic browser exceptions, Web Vitals, and
network timing. Handled game-worker failures are also reported explicitly with
the backend, model, deployment, and anonymous game ID when available. Network
request and response headers and request bodies are recorded in session replay.
Response bodies are removed before replay events are sent. Session replay
sampling is controlled remotely in PostHog rather than hard-coded in the client.

Game telemetry uses three custom events:

- `game_started` records the four color assignments, AlphaBlokus strength and
  rollout count for each bot, inference backend, model, and a random anonymous
  game ID.
- `game_move_played` records the game ID, ordered move number, canonical move
  index, player color and agent settings, and occupied board coordinates.
- `game_completed` records the winner, all four scores, duration, and the full
  ordered canonical move-index sequence. That sequence can replay the game
  exactly against the same move table.

The Pages build sets `VITE_APP_VERSION` to the deployment commit SHA so events
remain attributable if the model, move table, or rules change later.
