/**
 * The documentation shell, in the shape shadcn/ui's docs use: a section list on
 * the left, one page of content in the middle, an "On this page" rail on the
 * right, and previous/next at the foot.
 *
 * One shell, two homes. The dashboard renders `<Docs />`, which drops the top
 * bar because the app already has one; the published site renders `<DocsSite />`,
 * which adds it. The content itself lives in ./content.tsx and is shared, so
 * there is never a second copy to keep in step.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeft, ArrowRight, Menu, Moon, Search, Sun, X } from "lucide-react"

import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { GROUPS, SECTIONS, type Section } from "@/pages/docs/content"

const REPO = "https://github.com/sohangujari/nero"

/* -------------------------------------------------------------- routing --- */

/** The section named by the URL hash, defaulting to the first.
 *
 *  A hash rather than a path: GitHub Pages serves static files, so a real route
 *  like /docs/skills would 404 on reload unless every page were emitted
 *  separately. The hash costs nothing, survives reload, and is shareable. */
function useHashSection(): [Section, (id: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "")
  const [id, setId] = useState(read)

  useEffect(() => {
    const onHash = () => setId(read())
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])

  const section = SECTIONS.find((s) => s.id === id) ?? SECTIONS[0]
  const go = useCallback((next: string) => {
    window.location.hash = next
    // The hash only moves the URL; the reader expects to land at the top of the
    // page they just chose.
    document.getElementById("docs-scroll")?.scrollTo({ top: 0 })
    window.scrollTo({ top: 0 })
  }, [])
  return [section, go]
}

/** Highlights the sub-heading nearest the top of the viewport.
 *
 *  Reads scroll position rather than using IntersectionObserver: in the
 *  dashboard the article scrolls inside a ScrollArea viewport rather than the
 *  window, so an observer would need that element as its root anyway — at which
 *  point a scroll handler over a handful of headings is shorter and easier to
 *  reason about. It copes with either scroll root, because the standalone site
 *  scrolls the window. */
function useActiveHeading(section: Section): string {
  const [active, setActive] = useState("")
  useEffect(() => {
    setActive(section.headings[0]?.id ?? "")
    const article = document.getElementById("docs-article")
    const scroller = article?.closest('[data-slot="scroll-area-viewport"]')
    const target: HTMLElement | Window = scroller instanceof HTMLElement ? scroller : window
    const onScroll = () => {
      const top = scroller instanceof HTMLElement ? scroller.getBoundingClientRect().top : 0
      let current = section.headings[0]?.id ?? ""
      for (const { id } of section.headings) {
        const heading = document.getElementById(id)
        // 120px, so a heading counts as "reached" a little before it lands
        // flush against the top edge — otherwise the highlight lags a section.
        if (heading && heading.getBoundingClientRect().top - top < 120) current = id
      }
      setActive(current)
    }
    onScroll()
    target.addEventListener("scroll", onScroll, { passive: true })
    return () => target.removeEventListener("scroll", onScroll)
  }, [section])
  return active
}

/* ------------------------------------------------------------- controls --- */

const THEME_KEY = "nero-docs-theme"

/** Light/dark for the published site. The dashboard is dark-only and never
 *  mounts this, so nothing there can be toggled out from under it. */
function ThemeToggle() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem(THEME_KEY) !== "light"
    } catch {
      return true // private windows and blocked storage still get a readable page
    }
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    try {
      localStorage.setItem(THEME_KEY, dark ? "dark" : "light")
    } catch {
      /* a remembered preference is a convenience, not a requirement */
    }
  }, [dark])
  return (
    <button
      type="button"
      aria-label={dark ? "Switch to light" : "Switch to dark"}
      onClick={() => setDark((value) => !value)}
      className="text-muted-foreground hover:text-foreground hover:bg-accent rounded-md p-2"
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  )
}

type Hit = { id: string; label: string; context: string }

/** Titles and sub-headings only. Indexing the prose would mean shipping it
 *  twice — once as JSX to render and once as text to match — for a nine-page
 *  site whose headings already name everything it covers. */
function useSearchIndex(): Hit[] {
  return useMemo(
    () =>
      SECTIONS.flatMap((section) => [
        { id: section.id, label: section.title, context: section.group },
        ...section.headings.map((heading) => ({
          id: `${section.id}#${heading.id}`,
          label: heading.label,
          context: section.title,
        })),
      ]),
    [],
  )
}

function SearchBox({ onPick }: { onPick: (id: string) => void }) {
  const [query, setQuery] = useState("")
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const index = useSearchIndex()

  useEffect(() => {
    const onAway = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", onAway)
    return () => document.removeEventListener("mousedown", onAway)
  }, [])

  const needle = query.trim().toLowerCase()
  const hits = needle
    ? index.filter((hit) => hit.label.toLowerCase().includes(needle)).slice(0, 8)
    : []

  const pick = (hit: Hit) => {
    const [section, heading] = hit.id.split("#")
    onPick(section)
    setQuery("")
    setOpen(false)
    // The section has to render before its heading exists to scroll to.
    if (heading) {
      requestAnimationFrame(() =>
        document.getElementById(heading)?.scrollIntoView({ block: "start" }),
      )
    }
  }

  return (
    <div ref={box} className="relative w-full max-w-xs">
      <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
      <input
        value={query}
        onChange={(event) => {
          setQuery(event.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && hits[0]) pick(hits[0])
          if (event.key === "Escape") setOpen(false)
        }}
        placeholder="Search documentation…"
        className="bg-muted/50 focus-visible:ring-ring/50 h-9 w-full rounded-md border pl-9 text-sm outline-none focus-visible:ring-[3px]"
      />
      {open && hits.length > 0 && (
        <ul className="bg-popover absolute top-11 right-0 left-0 z-50 overflow-hidden rounded-md border py-1 shadow-md">
          {hits.map((hit) => (
            <li key={hit.id}>
              <button
                type="button"
                onClick={() => pick(hit)}
                className="hover:bg-accent flex w-full items-baseline gap-2 px-3 py-1.5 text-left text-sm"
              >
                <span>{hit.label}</span>
                <span className="text-muted-foreground ml-auto text-xs">{hit.context}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* --------------------------------------------------------------- layout --- */

function SectionNav({
  current,
  go,
  onNavigate,
}: {
  current: string
  go: (id: string) => void
  onNavigate?: () => void
}) {
  return (
    <nav className="space-y-6">
      {GROUPS.map((group) => (
        <div key={group}>
          <p className="mb-2 text-sm font-medium">{group}</p>
          <ul className="space-y-0.5">
            {SECTIONS.filter((section) => section.group === group).map((section) => (
              <li key={section.id}>
                <button
                  type="button"
                  onClick={() => {
                    go(section.id)
                    onNavigate?.()
                  }}
                  className={cn(
                    "w-full rounded-md px-3 py-1.5 text-left text-sm transition-colors",
                    current === section.id
                      ? "bg-accent text-accent-foreground font-medium"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {section.title}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function Body({ embedded }: { embedded: boolean }) {
  const [section, go] = useHashSection()
  const active = useActiveHeading(section)
  const [menu, setMenu] = useState(false)

  const order = SECTIONS.findIndex((s) => s.id === section.id)
  const previous = SECTIONS[order - 1]
  const next = SECTIONS[order + 1]

  return (
    <>
      {!embedded && (
        <header className="bg-background/80 sticky top-0 z-40 border-b backdrop-blur">
          <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4 sm:px-6">
            <button
              type="button"
              aria-label="Sections"
              onClick={() => setMenu(true)}
              className="text-muted-foreground hover:text-foreground -ml-1 rounded-md p-2 lg:hidden"
            >
              <Menu className="size-4" />
            </button>
            <a href="#introduction" className="flex items-center gap-2">
              <span className="bg-primary text-primary-foreground flex size-7 items-center justify-center rounded-lg text-xs font-semibold">
                N
              </span>
              <span className="font-semibold">Nero</span>
            </a>
            <span className="text-muted-foreground hidden text-sm sm:inline">Docs</span>
            <div className="ml-auto flex items-center gap-2">
              <div className="hidden sm:block">
                <SearchBox onPick={go} />
              </div>
              <a
                href={REPO}
                className="text-muted-foreground hover:text-foreground hover:bg-accent rounded-md px-2 py-1.5 text-sm"
              >
                GitHub
              </a>
              <ThemeToggle />
            </div>
          </div>
        </header>
      )}

      {/* Off-canvas section list for narrow screens. */}
      {menu && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="bg-background/80 absolute inset-0" onClick={() => setMenu(false)} />
          <div className="bg-background absolute inset-y-0 left-0 w-72 overflow-y-auto border-r p-5">
            <button
              type="button"
              aria-label="Close"
              onClick={() => setMenu(false)}
              className="text-muted-foreground hover:text-foreground mb-4 rounded-md p-1"
            >
              <X className="size-4" />
            </button>
            <SectionNav current={section.id} go={go} onNavigate={() => setMenu(false)} />
          </div>
        </div>
      )}

      <div className="mx-auto flex max-w-7xl gap-8 px-4 sm:px-6">
        <aside className="hidden w-52 shrink-0 py-8 lg:block">
          <div className={embedded ? "" : "sticky top-20"}>
            <SectionNav current={section.id} go={go} />
          </div>
        </aside>

        <article id="docs-article" className="min-w-0 flex-1 py-8 pb-20">
          <h1 className="text-3xl font-semibold tracking-tight">{section.title}</h1>
          <p className="text-muted-foreground mt-2 text-base leading-relaxed">{section.lede}</p>
          <div className="mt-6 space-y-3 text-sm leading-relaxed">{section.body}</div>

          <div className="mt-12 flex gap-3 border-t pt-6">
            {previous && (
              <button
                type="button"
                onClick={() => go(previous.id)}
                className="hover:bg-accent flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
              >
                <ArrowLeft className="size-3.5" />
                {previous.title}
              </button>
            )}
            {next && (
              <button
                type="button"
                onClick={() => go(next.id)}
                className="hover:bg-accent ml-auto flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
              >
                {next.title}
                <ArrowRight className="size-3.5" />
              </button>
            )}
          </div>
        </article>

        <aside className="hidden w-52 shrink-0 py-8 xl:block">
          <div className={embedded ? "sticky top-1" : "sticky top-20"}>
            <p className="mb-3 text-sm font-medium">On this page</p>
            <ul className="space-y-2 text-sm">
              {section.headings.map((heading) => (
                <li key={heading.id}>
                  <a
                    href={`#${section.id}`}
                    onClick={(event) => {
                      event.preventDefault()
                      document
                        .getElementById(heading.id)
                        ?.scrollIntoView({ behavior: "smooth", block: "start" })
                    }}
                    className={cn(
                      "hover:text-foreground block transition-colors",
                      active === heading.id
                        ? "text-foreground font-medium"
                        : "text-muted-foreground",
                    )}
                  >
                    {heading.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </>
  )
}

/** The dashboard's Docs page: no top bar, and it scrolls in its own pane. */
export function Docs() {
  return (
    <ScrollArea className="h-full" id="docs-scroll">
      <Body embedded />
    </ScrollArea>
  )
}

/** The published site: top bar, window scrolling, and a footer. */
export function DocsSite() {
  return (
    <>
      <Body embedded={false} />
      <footer className="text-muted-foreground mx-auto max-w-7xl px-4 pb-10 text-xs sm:px-6">
        Nero is a personal AI assistant. Source on{" "}
        <a href={REPO} className="underline underline-offset-4">
          GitHub
        </a>
        .
      </footer>
    </>
  )
}
