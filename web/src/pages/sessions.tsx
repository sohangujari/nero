import { Remove } from "@/components/editable"
import { Empty, Page, when } from "@/components/page"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { Edit, State } from "@/lib/api"

/** Session ids carry where the turn came from, so the list can say it. */
function origin(id: string): string {
  if (id.startsWith("telegram")) return "telegram"
  if (id.startsWith("voice") || id.startsWith("talk")) return "voice"
  if (id.startsWith("dashboard") || id.startsWith("web")) return "dashboard"
  if (id.startsWith("routine")) return "routine"
  return "terminal"
}

export function Sessions({ state, edit }: { state: State; edit: Edit }) {
  return (
    <Page
      title="Sessions"
      lede="Every conversation in the transcript, newest first. All channels write to the same store, so a thing said on the phone is remembered at the terminal — and deleting one here removes it from recall everywhere."
    >
      {state.sessions.length === 0 ? (
        <Empty>Nothing recorded yet.</Empty>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Session</TableHead>
              <TableHead>Started</TableHead>
              <TableHead>Last message</TableHead>
              <TableHead className="text-right">Messages</TableHead>
              <TableHead className="w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {state.sessions.map((session) => (
              <TableRow key={session.session_id}>
                <TableCell>
                  <span className="font-mono text-xs [overflow-wrap:anywhere]">
                    {session.session_id}
                  </span>
                  <Badge variant="outline" className="ml-2">
                    {origin(session.session_id)}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground text-xs">
                  {when(session.started_at)}
                </TableCell>
                <TableCell className="text-muted-foreground text-xs">
                  {when(session.last_at)}
                </TableCell>
                <TableCell className="text-right text-sm tabular-nums">{session.turns}</TableCell>
                <TableCell>
                  <Remove
                    what="this conversation"
                    detail={`${session.turns} message(s) are deleted from the transcript, the keyword index and semantic recall. This cannot be undone.`}
                    onConfirm={() => edit("forget_session", session.session_id)}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Page>
  )
}
