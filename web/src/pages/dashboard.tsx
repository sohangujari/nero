import { ArrowRight } from "lucide-react"

import type { PageId } from "@/components/app-sidebar"
import { Page, when } from "@/components/page"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { State } from "@/lib/api"

function Stat({
  label,
  value,
  hint,
  onClick,
}: {
  label: string
  value: string | number
  hint?: string
  onClick?: () => void
}) {
  return (
    <Card
      onClick={onClick}
      className={onClick ? "hover:bg-accent/40 cursor-pointer transition-colors" : undefined}
    >
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl [overflow-wrap:anywhere]">{value}</CardTitle>
      </CardHeader>
      {hint && <CardContent className="text-muted-foreground pt-0 text-xs">{hint}</CardContent>}
    </Card>
  )
}

export function Dashboard({ state, go }: { state: State; go: (id: PageId) => void }) {
  const { counts, models, channels, sessions, memory } = state
  const open = Object.entries(channels).filter(([, c]) => c.enabled)
  const last = sessions[0]

  return (
    <Page
      title="Dashboard"
      lede={`${state.assistant} is running here, in ${state.mode} mode. Everything below is read live from this session.`}
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Stat
          label="Answering with"
          value={models.model}
          hint={`${models.provider}${models.fallback_chain.length ? ` · ${models.fallback_chain.length} fallback(s)` : ""}`}
          onClick={() => go("models")}
        />
        <Stat
          label="Skills available"
          value={`${counts.skills_available} / ${counts.skills_total}`}
          hint="Enabled, and allowed by the current mode"
          onClick={() => go("skills")}
        />
        <Stat
          label="Channels open"
          value={open.length}
          hint={open.map(([name]) => name).join(", ")}
          onClick={() => go("channels")}
        />
        <Stat
          label="Conversations"
          value={counts.sessions}
          hint={`${counts.turns} messages · last ${when(last?.last_at)}`}
          onClick={() => go("sessions")}
        />
        <Stat
          label="Routines"
          value={counts.routines}
          hint={counts.routines ? "Scheduled prompts" : "None scheduled"}
          onClick={() => go("routines")}
        />
        <Stat
          label="Remembered facts"
          value={memory.facts}
          hint={memory.semantic_recall ? "Recall: keyword + semantic" : "Recall: keyword only"}
          onClick={() => go("memory")}
        />
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-base">Ask something</CardTitle>
          <CardDescription>
            A turn here is the same turn as one in the terminal — same model, same memory, same
            skills.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={() => go("chat")}>
            Open chat <ArrowRight />
          </Button>
        </CardContent>
      </Card>

      <div className="mt-6 flex flex-wrap gap-2">
        <Badge variant="outline">mode: {state.mode}</Badge>
        <Badge variant="outline">routing: {models.route_by}</Badge>
        {counts.mcp > 0 && <Badge variant="outline">{counts.mcp} MCP server(s)</Badge>}
      </div>
    </Page>
  )
}
