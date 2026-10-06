# Hosting and demo options

## Recorded browser showcase

`docs/index.html` is a static showcase with the existing demo recording, a pipeline overview, and links to the source and model release. It does not need an API key, microphone access, or a running model server. It embeds the publicly accessible Google Drive recording, with a direct viewer link as a fallback. Keep the Drive file shared with anyone who has the link.

Preview locally from the repository root:

```bash
python -m http.server 8080 --bind 127.0.0.1
```

Open http://localhost:8080/docs/. This starts only a static file server, not the agent.

After approval to publish, GitHub Pages can serve the `docs` directory from `main`. In repository Settings ? Pages, choose ?Deploy from a branch?, `main`, and `/docs`. Expected URL: https://rababb-p.github.io/CustomVoiceAgent/. This URL is a publishing target, not a claim that the site is live. See [GitHub's publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

No deployment workflow or remote Pages settings have been added or changed in this draft. A direct Drive viewer link remains available if the embedded player cannot load the video. Captions/transcripts have not yet been added to the existing recording.

## Interactive voice hosting

GitHub Pages can serve the showcase but cannot run the Python voice backend. An interactive deployment needs a persistent Python process with sufficient memory/disk for Whisper, embeddings, and Kokoro, plus WebSocket support. Choose the host after measuring startup memory and CPU latency locally.

Before exposing the existing app:

1. Provide Gemini credentials as server-side secrets and verify model access.
2. Index only deliberately public corpus content; provision model files and persistent storage.
3. Serve HTTPS and change the browser WebSocket URL to select `wss:` on HTTPS. The current client hardcodes `ws:`. Browsers require a secure context for microphone access; localhost is suitable for development. See [MDN's microphone API documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).
4. Add access controls, per-user usage/concurrency limits, audio size/duration limits, and request timeouts. The current client-side model RPM limiter is not a public-user quota system.
5. Define cache/log retention and display a notice that transcripts and retrieved context may be sent to Gemini.
6. Test microphone permission failures, long recordings, disconnects, concurrent visitors, model initialization, and provider quota failures.

These are requirements for a future interactive deployment, not capabilities implemented by this documentation change. A measured, controlled preview should come before public launch.
