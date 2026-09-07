import { SendHorizontal } from "lucide-react"
import { useEffect, useRef } from "react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export function Composer({
  value,
  onChange,
  onSubmit,
  busy,
}: {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
}) {
  const box = useRef<HTMLTextAreaElement>(null)

  // Grow with the text instead of scrolling a two-line window, capped so a
  // pasted essay cannot push the conversation off screen.
  useEffect(() => {
    const el = box.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  useEffect(() => {
    if (!busy) box.current?.focus()
  }, [busy])

  return (
    <form
      className="flex items-end gap-2"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <Textarea
        ref={box}
        rows={1}
        value={value}
        placeholder="Ask anything…"
        className="max-h-[200px] min-h-0 resize-none py-2.5"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            onSubmit()
          }
        }}
      />
      <Button type="submit" size="icon" disabled={busy || !value.trim()} aria-label="Send">
        <SendHorizontal className="size-4" />
      </Button>
    </form>
  )
}
