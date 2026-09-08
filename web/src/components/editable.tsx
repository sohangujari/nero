import { Trash2 } from "lucide-react"
import { useEffect, useState, type ReactNode } from "react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import type { Edit } from "@/lib/api"

/**
 * The four controls every editable page is built from. Each one names the
 * config key it writes, which is the same dotted key you would type at
 * `nero config set` — so the page and the CLI can never mean different things
 * by the same setting.
 *
 * None of them keep their own idea of the value once saved: the server answers
 * an edit with the whole new state, and the page re-renders from that. A
 * control that trusted its own optimistic value would happily display a
 * setting the validator had coerced or refused.
 */

export function Toggle({
  field,
  on,
  edit,
  label,
}: {
  field: string
  on: boolean
  edit: Edit
  label?: string
}) {
  const [busy, setBusy] = useState(false)
  return (
    <Switch
      checked={on}
      disabled={busy}
      aria-label={label ?? field}
      onCheckedChange={async (next) => {
        setBusy(true)
        await edit("set", field, String(next))
        setBusy(false)
      }}
    />
  )
}

/** Commits on Enter or blur, never per keystroke — every save is a validated
 *  round trip, and one per character would be both slow and wrong. */
export function TextField({
  field,
  value,
  edit,
  placeholder,
  className,
}: {
  field: string
  value: string
  edit: Edit
  placeholder?: string
  className?: string
}) {
  const [draft, setDraft] = useState(value)
  const [busy, setBusy] = useState(false)

  // The server is the source of truth: if a save coerced the value, or another
  // edit changed it, the box follows.
  useEffect(() => setDraft(value), [value])

  const commit = async () => {
    if (draft === value || busy) return
    setBusy(true)
    await edit("set", field, draft)
    setBusy(false)
  }

  return (
    <Input
      value={draft}
      disabled={busy}
      placeholder={placeholder ?? "—"}
      className={className ?? "h-8 max-w-sm font-mono text-xs"}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur()
        if (event.key === "Escape") setDraft(value)
      }}
    />
  )
}

export function SelectField({
  field,
  value,
  options,
  edit,
}: {
  field: string
  value: string
  options: string[]
  edit: Edit
}) {
  const [busy, setBusy] = useState(false)
  return (
    <Select
      value={value}
      disabled={busy}
      onValueChange={async (next) => {
        setBusy(true)
        await edit("set", field, next)
        setBusy(false)
      }}
    >
      <SelectTrigger className="h-8 w-56 font-mono text-xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option} value={option} className="font-mono text-xs">
            {option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** Destructive actions ask first. The dialog states what is being deleted and
 *  whether it can be undone, because from here it cannot. */
export function Remove({
  what,
  detail,
  onConfirm,
  children,
}: {
  what: string
  detail: ReactNode
  onConfirm: () => Promise<void> | void
  children?: ReactNode
}) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        {children ?? (
          <Button variant="ghost" size="icon" className="size-8" aria-label={`Remove ${what}`}>
            <Trash2 className="text-muted-foreground size-4" />
          </Button>
        )}
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Remove {what}?</AlertDialogTitle>
          <AlertDialogDescription>{detail}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => void onConfirm()}
            className="bg-destructive text-white hover:bg-destructive/90"
          >
            Remove
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
