# Runtime Architecture

Qwen Image Studio runs as a single Kaggle notebook cell that bootstraps a temporary inference service and browser UI.

```mermaid
flowchart LR
    subgraph Kaggle[ Kaggle GPU session ]
        N[Notebook cell] -->|install packages| P[Python environment]
        N -->|write and launch| A[app.py child process]
        A --> G[GPU and CUDA check]
        G --> M[QwenImage21Pipeline]
        M --> Q[4 bit NF4 transformer and text encoder]
        M --> V[FP16 VAE with tiling and slicing]
        A --> UI[Gradio Blocks UI]
        UI -->|prompt, size, steps, seed| L[Generation lock]
        L --> M
        M -->|PIL image| S[PNG in generated_images]
        M -->|image and run status| UI
    end
    UI -->|temporary share tunnel| B[User browser]
```

## Components

1. **Notebook bootstrap** removes selected conflicting libraries, installs model and UI dependencies, creates `/kaggle/working/generated_images`, writes `/kaggle/working/app.py`, and streams the child process output.
2. **Model service** checks CUDA and loads `Qwen/Qwen-Image-2.1` using Diffusers' `QwenImage21Pipeline`. The transformer and text encoder use bitsandbytes 4 bit NF4 quantization with FP16 compute. The VAE remains FP16, with tiling and slicing enabled when supported.
3. **Gradio UI** offers prompt, width, height, inference steps, and seed controls. The service clamps dimensions to 512–1024 and steps to 10–40. A nonblocking lock rejects a second concurrent generation.
4. **Output path** saves each image as a timestamp and seed named PNG under `/kaggle/working/generated_images`. This is temporary Kaggle working storage.
5. **Access path** starts Gradio with `share=True`, producing a temporary public tunnel URL. The URL and running app depend on the Kaggle cell and session staying alive.

## Sequence

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant App as app.py
    participant Model as Qwen pipeline on GPU
    participant Disk as Kaggle working storage

    User->>Notebook: Run the cell
    Notebook->>Notebook: Install dependencies and create app.py
    Notebook->>App: Start child process
    App->>Model: Download and load model once
    App-->>User: Print temporary Gradio URL
    User->>App: Submit prompt and generation settings
    App->>Model: Generate under inference mode
    Model-->>App: Return image
    App->>Disk: Save PNG
    App-->>User: Display image and status
```

## Operational boundaries

- Internet is required for dependency installation and model download.
- A CUDA GPU is required. The recorded test used a Tesla T4; actual speed and available VRAM vary.
- The notebook reinstalls selected libraries and installs the latest Diffusers from GitHub, so dependency behavior can change over time.
- There is no built-in prompt moderation or safety filter.
- The temporary share link should be treated as public while active.
