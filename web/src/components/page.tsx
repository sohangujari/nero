import type { ReactNode } from "react"

import { ScrollArea } from "@/components/ui/scroll-area"

/** Every page but Chat is a scrolling read-only sheet with the same header,
 *  so the shell lives here once rather than in each of them. */
export function Page({
  title,
  lede,
  children,
}: {
  title: string
  lede: string
  children: ReactNode
}) {
  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-4xl px-6 py-6">
        <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
        <p className="text-muted-foreground mt-1 mb-6 text-sm">{lede}</p>
        {children}
      </div>
    </ScrollArea>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="text-muted-foreground rounded-lg border border-dashed px-6 py-10 text-center text-sm">
      {children}
    </div>
  )
}

/** A label/value row. Values are rendered monospace because most of them are
 *  identifiers a person will retype at the command line. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-4 border-b py-2 last:border-0">
      <dt className="text-muted-foreground w-56 shrink-0 text-xs">{label}</dt>
      <dd className="font-mono text-sm [overflow-wrap:anywhere]">{children}</dd>
    </div>
  )
}

export function list(values: string[] | null | undefined): string {
  return values && values.length ? values.join(", ") : "—"
}

export function when(value: string | undefined | null): string {
  if (!value) return "—"
  const parsed = new Date(value)
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString()
}
