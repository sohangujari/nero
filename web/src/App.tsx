import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { Activity } from "@/components/activity"
import { Composer } from "@/components/composer"
import { MessageRow, type Message } from "@/components/message"
import { Settings } from "@/components/settings"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { NeroClient, readToken } from "@/lib/api"

let nextId = 0
const message = (m: Omit<Message, "id">): Message => ({ id: nextId++, ...m })

export default function App() {
  const [token] = useState(readToken)
  const [name, setName] = useState("Nero")
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)

  // Memoised: Activity and Settings take it as an effect dependency, and a new
  // object every render would refetch on every keystroke.
  const client = useMemo(() => (token ? new NeroClient(token) : null), [token])

  useEffect(() => {
    if (!client) return
    client.name().then((n) => {
      setName(n)
      document.title = n
    })
    client
      .history()
      .then((turns) =>
        setMessages(turns.map((t) => message({ role: t.role, content: t.content }))),
      )
      .catch(() => {
        /* an empty window is a fine starting point */
      })
    // The client is derived from a token that never changes for a session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = useCallback(async () => {
    const text = draft.trim()
    if (!text || busy || !client) return
    setDraft("")
    setBusy(true)
    const pending = message({ role: "assistant", content: "Thinking…", pending: true })
    setMessages((prev) => [...prev, message({ role: "user", content: text }), pending])
    try {
      const reply = await client.ask(text)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pending.id ? { ...m, content: reply || "(no reply)", pending: false } : m,
        ),
      )
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not reach Nero."
      setMessages((prev) =>
        prev.map((m) => (m.id === pending.id ? { ...m, content: detail, pending: false, error: true } : m)),
      )
    } finally {
      setBusy(false)
    }
  }, [draft, busy, client])

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

  return (
    <Tabs defaultValue="chat" className="mx-auto flex h-svh max-w-3xl flex-col gap-0">
      <header className="flex items-center gap-3 px-5 py-3">
        <h1 className="text-base font-semibold tracking-tight">{name}</h1>
        <span className="text-muted-foreground text-xs">local · this browser only</span>
        <TabsList className="ml-auto">
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>
      </header>
      <Separator />

      <TabsContent value="chat" className="flex min-h-0 flex-1 flex-col">
        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-5 px-5 py-6">
            {messages.length === 0 && (
              <p className="text-muted-foreground py-16 text-center text-sm">
                Nothing yet. Ask anything.
              </p>
            )}
            {messages.map((m) => (
              <MessageRow key={m.id} message={m} />
            ))}
            <div ref={bottom} />
          </div>
        </ScrollArea>
        <Separator />
        <div className="px-5 py-4">
          <Composer value={draft} onChange={setDraft} onSubmit={send} busy={busy} />
        </div>
      </TabsContent>

      <TabsContent value="activity" className="min-h-0 flex-1">
        <Activity client={client!} />
      </TabsContent>

      <TabsContent value="config" className="min-h-0 flex-1">
        <Settings client={client!} />
      </TabsContent>
    </Tabs>
  )
}
