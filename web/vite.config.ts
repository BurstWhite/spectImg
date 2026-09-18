import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // dev only: forward API calls to the local uvicorn server
      '/api': 'http://localhost:8000',
    },
  },
})
