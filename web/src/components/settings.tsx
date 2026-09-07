import { useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { NeroClient } from "@/lib/api"

/** Flatten nested config into "a.b.c" rows — the shape people already type at
 *  `nero config set`, so what they read here matches what they would write. */
function rows(value: unknown, prefix = ""): [string, string][] {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    const text = Array.isArray(value)
      ? value.length
        ? value.join(", ")
        : "—"
      : value === null
        ? "—"
        : String(value)
    return [[prefix, text]]
  }
  return Object.entries(value as Record<string, unknown>).flatMap(([key, inner]) =>
    rows(inner, prefix ? `${prefix}.${key}` : key),
  )
}

export function Settings({ client }: { client: NeroClient }) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    client.config().then(setConfig).catch(() => setConfig({}))
  }, [client])

  if (config === null) {
    return <p className="text-muted-foreground p-6 text-sm">Loading…</p>
  }
  const entries = rows(config).filter(([key]) => key)
  return (
    <ScrollArea className="h-full">
      <div className="px-5 py-4">
        <p className="text-muted-foreground mb-4 text-xs">
          Read-only. API keys live in your OS keyring and are never sent here.
          Change anything with <code className="text-foreground">nero config</code>.
        </p>
        <dl className="divide-border divide-y text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="flex items-baseline gap-4 py-2">
              <dt className="text-muted-foreground w-64 shrink-0 font-mono text-xs">{key}</dt>
              <dd className="[overflow-wrap:anywhere]">
                {value === "true" || value === "false" ? (
                  <Badge variant={value === "true" ? "secondary" : "outline"}>{value}</Badge>
                ) : (
                  value
                )}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </ScrollArea>
  )
}
