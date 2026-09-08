import { useCallback, useEffect, useRef, useState } from "react"

import { Composer } from "@/components/composer"
import { MessageRow, type Message } from "@/components/message"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import type { NeroClient } from "@/lib/api"

let nextId = 0
const message = (m: Omit<Message, "id">): Message => ({ id: nextId++, ...m })

export function Chat({ client }: { client: NeroClient }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    client
      .history()
      .then((turns) =>
        setMessages(turns.map((t) => message({ role: t.role, content: t.content }))),
      )
      .catch(() => {
        /* an empty window is a fine starting point */
      })
  }, [client])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const send = useCallback(async () => {
    const text = draft.trim()
    if (!text || busy) return
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
        prev.map((m) =>
          m.id === pending.id ? { ...m, content: detail, pending: false, error: true } : m,
        ),
      )
    } finally {
      setBusy(false)
    }
  }, [draft, busy, client])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ScrollArea className="flex-1">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-6">
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
      <div className="mx-auto w-full max-w-3xl px-6 py-4">
        <Composer value={draft} onChange={setDraft} onSubmit={send} busy={busy} />
      </div>
    </div>
  )
}
