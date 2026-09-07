/**
 * The token Nero printed with the URL.
 *
 * Kept in sessionStorage and stripped from the address bar: leaving it in the
 * URL means it rides into browser history and into the Referer of anything the
 * page ever links to. sessionStorage is per-tab and dies with it, so a reload
 * still works but a new tab has to be opened from the printed link again.
 */
const KEY = "nero-token"

export function readToken(): string | null {
  const fromUrl = new URLSearchParams(window.location.search).get("token")
  if (fromUrl) {
    try {
      sessionStorage.setItem(KEY, fromUrl)
    } catch {
      /* private mode: fall through, the in-memory value below still serves */
    }
    window.history.replaceState({}, "", window.location.pathname)
    return fromUrl
  }
  try {
    return sessionStorage.getItem(KEY)
  } catch {
    return null
  }
}

export type Turn = { role: "user" | "assistant"; content: string }

export type AuditEntry = {
  id?: number
  skill?: string
  outcome?: string
  provider?: string
  arguments?: unknown
  created_at?: string
  [key: string]: unknown
}

export class NeroClient {
  constructor(private token: string) {}

  private headers(): HeadersInit {
    return { "Content-Type": "application/json", "X-Nero-Token": this.token }
  }

  async name(): Promise<string> {
    const response = await fetch("/api/meta", { headers: this.headers() })
    if (!response.ok) return "Nero"
    return (await response.json()).name ?? "Nero"
  }

  async audit(): Promise<AuditEntry[]> {
    const response = await fetch("/api/audit", { headers: this.headers() })
    if (!response.ok) return []
    return response.json()
  }

  async config(): Promise<Record<string, unknown>> {
    const response = await fetch("/api/config", { headers: this.headers() })
    if (!response.ok) return {}
    return response.json()
  }

  async history(): Promise<Turn[]> {
    const response = await fetch("/api/history", { headers: this.headers() })
    if (!response.ok) return []
    return response.json()
  }

  async ask(text: string, signal?: AbortSignal): Promise<string> {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ text }),
      signal,
    })
    const data = await response.json().catch(() => null)
    if (!response.ok || !data) {
      throw new Error(data?.error ?? `Nero answered with HTTP ${response.status}.`)
    }
    return data.reply as string
  }
}
