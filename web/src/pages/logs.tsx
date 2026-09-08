import { useEffect, useState } from "react"

import { Empty, Page, when } from "@/components/page"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import type { AuditEntry, NeroClient } from "@/lib/api"

/** The audit log records what a skill was asked to do; showing it is most of
 *  the point of keeping one. Failures are worth spotting from across the room. */
function failed(entry: AuditEntry): boolean {
  return /^(error|failed|refused|denied|blocked)/i.test(entry.result_summary ?? "")
}

function args(entry: AuditEntry): string {
  const pairs = Object.entries(entry.arguments ?? {})
  if (pairs.length === 0) return ""
  return pairs
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ·  ")
}

export function Logs({ client }: { client: NeroClient }) {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null)

  useEffect(() => {
    client
      .audit()
      .then(setEntries)
      .catch(() => setEntries([]))
  }, [client])

  return (
    <Page
      title="Logs"
      lede="Every skill Nero ran, most recent first — the ground truth when a model claims it did something it did not."
    >
      {entries === null ? (
        <div className="space-y-3">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : entries.length === 0 ? (
        <Empty>Nothing recorded yet. Every skill Nero runs shows up here.</Empty>
      ) : (
        <ul className="divide-border divide-y">
          {entries.map((entry, i) => (
            <li key={i} className="flex flex-col gap-1 py-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs font-medium">{entry.skill_name}</span>
                {failed(entry) && <Badge variant="destructive">failed</Badge>}
                <Badge variant="outline">{entry.provider}</Badge>
                <span className="text-muted-foreground ml-auto text-xs">
                  {when(entry.timestamp)}
                </span>
              </div>
              {args(entry) && (
                <p className="text-muted-foreground font-mono text-xs [overflow-wrap:anywhere]">
                  {args(entry)}
                </p>
              )}
              <p className="[overflow-wrap:anywhere]">{entry.result_summary}</p>
            </li>
          ))}
        </ul>
      )}
    </Page>
  )
}
