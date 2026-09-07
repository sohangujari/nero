import { useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { AuditEntry, NeroClient } from "@/lib/api"

function when(value: unknown): string {
  if (typeof value !== "string") return ""
  const parsed = new Date(value)
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString()
}

/** Anything the audit log records that this view has no column for. */
function extras(entry: AuditEntry): string {
  const known = new Set(["id", "skill", "outcome", "provider", "created_at", "requested_at"])
  const rest = Object.entries(entry).filter(([k, v]) => !known.has(k) && v !== null && v !== "")
  return rest.map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join("  ·  ")
}

export function Activity({ client }: { client: NeroClient }) {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null)

  useEffect(() => {
    client.audit().then(setEntries).catch(() => setEntries([]))
  }, [client])

  if (entries === null) {
    return <p className="text-muted-foreground p-6 text-sm">Loading…</p>
  }
  if (entries.length === 0) {
    return (
      <p className="text-muted-foreground p-6 text-sm">
        Nothing recorded yet. Every skill Nero runs shows up here.
      </p>
    )
  }
  return (
    <ScrollArea className="h-full">
      <ul className="divide-border divide-y">
        {entries.map((entry, i) => {
          const failed = entry.outcome && entry.outcome !== "ok" && entry.outcome !== "success"
          return (
            <li key={entry.id ?? i} className="flex flex-col gap-1 px-5 py-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{entry.skill ?? "—"}</span>
                {entry.outcome && (
                  <Badge variant={failed ? "destructive" : "secondary"}>{entry.outcome}</Badge>
                )}
                {entry.provider && (
                  <span className="text-muted-foreground text-xs">{entry.provider}</span>
                )}
                <span className="text-muted-foreground ml-auto text-xs">
                  {when(entry.created_at ?? entry.requested_at)}
                </span>
              </div>
              {extras(entry) && (
                <p className="text-muted-foreground text-xs [overflow-wrap:anywhere]">
                  {extras(entry)}
                </p>
              )}
            </li>
          )
        })}
      </ul>
    </ScrollArea>
  )
}
