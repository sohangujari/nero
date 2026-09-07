import { Bot, TriangleAlert, User } from "lucide-react"

import { cn } from "@/lib/utils"

export type Message = {
  id: number
  role: "user" | "assistant"
  content: string
  error?: boolean
  pending?: boolean
}

export function MessageRow({ message }: { message: Message }) {
  const isUser = message.role === "user"
  const Icon = message.error ? TriangleAlert : isUser ? User : Bot
  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full border",
          message.error
            ? "border-destructive/40 text-destructive"
            : isUser
              ? "bg-primary text-primary-foreground border-transparent"
              : "bg-muted text-muted-foreground",
        )}
      >
        <Icon className="size-4" />
      </div>
      <div
        className={cn(
          "max-w-[min(42rem,80%)] rounded-lg px-3.5 py-2.5 text-sm leading-relaxed",
          // Replies arrive as plain text, newlines and all; pre-wrap keeps the
          // model's own paragraphing without pulling in a markdown renderer.
          "whitespace-pre-wrap [overflow-wrap:anywhere]",
          isUser && "bg-primary text-primary-foreground",
          !isUser && !message.error && "bg-muted",
          message.error && "border-destructive/40 text-destructive border",
          message.pending && "text-muted-foreground italic",
        )}
      >
        {message.content}
      </div>
    </div>
  )
}
