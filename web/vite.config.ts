// import.meta.dirname keeps Vite's native config loader happy.
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
  // Built straight into the Python package so `pip install nero` and the
  // PyInstaller binary both carry the UI with no node on the user's machine.
  build: { outDir: "../nero/webui_dist", emptyOutDir: true },
})
