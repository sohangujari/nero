/**
 * Entry point for the published documentation site (GitHub Pages).
 *
 * The dashboard and the site render the same sections from
 * `@/pages/docs/content`, so there is one copy of the content. The only
 * difference is the frame: no app sidebar, no token, no API — the page is
 * static, which is what makes it publishable at all.
 */
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import { DocsSite } from "@/pages/docs"
import "@/index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DocsSite />
  </StrictMode>,
)
