import { useCallback, useEffect, useMemo, useState } from "react"
import { toast } from "sonner"

import { AppSidebar, PAGES, type PageId } from "@/components/app-sidebar"
import { Empty, Page } from "@/components/page"
import { Card } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { Toaster } from "@/components/ui/sonner"
import { Chat } from "@/pages/chat"
import { Config } from "@/pages/config"
import { Dashboard } from "@/pages/dashboard"
import { Docs } from "@/pages/docs"
import { Logs } from "@/pages/logs"
import { Sessions } from "@/pages/sessions"
import { Channels, Mcp, Memory, Models, Routines, Skills } from "@/pages/setup"
import { NeroClient, readToken, type Edit, type State } from "@/lib/api"

export default function App() {
  const [token] = useState(readToken)
  const [page, setPage] = useState<PageId>("dashboard")
  const [state, setState] = useState<State | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [version, setVersion] = useState(0)

  // Memoised: the pages take it as an effect dependency, and a new object
  // every render would refetch on every keystroke.
  const client = useMemo(() => (token ? new NeroClient(token) : null), [token])

  // One fetch for every read-only page, refreshed when the user comes back to
  // the Dashboard. Edits refresh it too — they answer with the new state.
  useEffect(() => {
    if (!client || page !== "dashboard") return
    client
      .state()
      .then((next) => {
        setState(next)
        setFailure(null)
        document.title = next.assistant
      })
      .catch((error: unknown) =>
        setFailure(error instanceof Error ? error.message : "Could not reach Nero."),
      )
  }, [client, page])

  /**
   * Every change on every page goes through here. The reply is the whole new
   * state, so the UI never shows a value Nero did not actually save — and a
   * rejected edit surfaces the validator's own words rather than a shrug.
   */
  const edit = useCallback<Edit>(
    async (action, key, value) => {
      if (!client) return
      try {
        setState(await client.edit(action, key, value))
        setVersion((n) => n + 1)
        toast.success(action === "set" ? `Saved ${key}` : `Removed ${key}`)
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Nero refused that change.")
      }
    },
    [client],
  )

  if (!token) {
    return (
      <div className="flex h-svh items-center justify-center p-6">
        <Card className="max-w-sm p-6 text-center text-sm">
          <p className="font-medium">This window needs the link Nero printed.</p>
          <p className="text-muted-foreground mt-2">
            Open the address from your terminal, token and all.
          </p>
        </Card>
      </div>
    )
  }

  const title = PAGES.find((p) => p.id === page)?.label ?? "Dashboard"

  return (
    <SidebarProvider>
      <AppSidebar
        name={state?.assistant ?? "Nero"}
        mode={state?.mode ?? "…"}
        page={page}
        onSelect={setPage}
        counts={{
          skills: state?.counts.skills_total,
          sessions: state?.counts.sessions,
          routines: state?.counts.routines,
          mcp: state?.counts.mcp,
        }}
      />
      <SidebarInset className="h-svh min-h-0 overflow-hidden">
        <header className="flex h-14 shrink-0 items-center gap-2 px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-1 h-4" />
          <span className="text-sm font-medium">{title}</span>
          <span className="text-muted-foreground ml-auto text-xs">local · this browser only</span>
        </header>
        <Separator />
        <div className="min-h-0 flex-1">
          {page === "chat" ? (
            <Chat client={client!} />
          ) : page === "docs" ? (
            <Docs />
          ) : page === "logs" ? (
            <Logs client={client!} />
          ) : page === "config" ? (
            <Config client={client!} edit={edit} version={version} />
          ) : failure ? (
            <Page title={title} lede="Nero could not be read.">
              <Empty>{failure}</Empty>
            </Page>
          ) : !state ? (
            <Page title={title} lede="Reading this session…">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-24 w-full" />
                ))}
              </div>
            </Page>
          ) : page === "dashboard" ? (
            <Dashboard state={state} go={setPage} />
          ) : page === "channels" ? (
            <Channels state={state} edit={edit} />
          ) : page === "sessions" ? (
            <Sessions state={state} edit={edit} />
          ) : page === "models" ? (
            <Models state={state} edit={edit} />
          ) : page === "skills" ? (
            <Skills state={state} edit={edit} />
          ) : page === "routines" ? (
            <Routines state={state} edit={edit} />
          ) : page === "mcp" ? (
            <Mcp state={state} edit={edit} />
          ) : (
            <Memory state={state} edit={edit} />
          )}
        </div>
      </SidebarInset>
      <Toaster position="bottom-right" />
    </SidebarProvider>
  )
}
