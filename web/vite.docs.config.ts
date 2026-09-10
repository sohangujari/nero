// Builds the public documentation site from the same components the dashboard
// uses. Separate from vite.config.ts because the two builds disagree on
// everything that matters: a different entry, a different output directory,
// and a relative base path, since the published site is served from /<repo>/
// rather than the domain root.
import { renameSync } from "node:fs"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig, type Plugin } from "vite"

const OUT = new URL("../site/", import.meta.url)

/** Vite names the output after its entry, so the build lands on docs.html.
 *  Pages serves index.html for a bare directory URL, so it has to be renamed —
 *  done here rather than in the npm script so the build needs no shell. */
function nameEntryIndex(): Plugin {
  return {
    name: "docs-entry-as-index",
    closeBundle() {
      renameSync(new URL("docs.html", OUT), new URL("index.html", OUT))
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), nameEntryIndex()],
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
  // Relative, so the built site works wherever it is served from: a project
  // page under /<repo>/, a user site or custom domain at the root, a local
  // preview, or straight off the filesystem. An absolute base has to name the
  // deploy path at build time and 404s everywhere else, which is exactly the
  // trap this avoids. Safe here because the site is a single page — there are
  // no deep URLs for "./" to resolve against differently.
  base: "./",
  build: {
    outDir: "../site",
    emptyOutDir: true,
    rollupOptions: { input: new URL("./docs.html", import.meta.url).pathname },
  },
})
