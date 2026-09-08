import { useEffect, useState } from "react"

import { TextField, Toggle } from "@/components/editable"
import { Page } from "@/components/page"
import { Skeleton } from "@/components/ui/skeleton"
import type { Edit, NeroClient } from "@/lib/api"

/** Flatten nested config into "a.b.c" rows — the shape people already type at
 *  `nero config set`, which is also the key each control writes. */
function rows(value: unknown, prefix = ""): [string, unknown][] {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return [[prefix, value]]
  }
  return Object.entries(value as Record<string, unknown>).flatMap(([key, inner]) =>
    rows(inner, prefix ? `${prefix}.${key}` : key),
  )
}

export function Config({
  client,
  edit,
  version,
}: {
  client: NeroClient
  edit: Edit
  version: number
}) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)

  // `version` bumps on every saved edit, so a change made on another page is
  // reflected here rather than leaving a stale value on screen.
  useEffect(() => {
    client
      .config()
      .then(setConfig)
      .catch(() => setConfig({}))
  }, [client, version])

  return (
    <Page
      title="Config"
      lede="Every setting, flattened to the keys you would type at `nero config set`. Edits save on Enter or when you click away, through the same validation the CLI uses. API keys are not here: they live in the OS keyring."
    >
      {config === null ? (
        <div className="space-y-2">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      ) : (
        <dl className="divide-border divide-y text-sm">
          {rows(config)
            .filter(([key]) => key)
            .map(([key, value]) => (
              <div key={key} className="flex items-center gap-4 py-2">
                <dt className="text-muted-foreground w-72 shrink-0 font-mono text-xs">{key}</dt>
                <dd className="flex-1">
                  {typeof value === "boolean" ? (
                    <Toggle field={key} on={value} edit={edit} label={key} />
                  ) : (
                    <TextField
                      field={key}
                      value={
                        Array.isArray(value)
                          ? value.join(", ")
                          : value === null
                            ? ""
                            : String(value)
                      }
                      edit={edit}
                      className="h-8 w-full max-w-xl font-mono text-xs"
                    />
                  )}
                </dd>
              </div>
            ))}
        </dl>
      )}
    </Page>
  )
}
