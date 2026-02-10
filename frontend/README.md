# Prosperity Grove Frontend

This UI is a console-style research interface for running simulations, inspecting metrics, and streaming live progress from the FastAPI backend.

## Local development
1. Install dependencies
```
npm install
```
2. Run the dev server
```
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

## Build
```
npm run build
```

## Environment variables
- `VITE_API_BASE` sets the FastAPI base URL.
