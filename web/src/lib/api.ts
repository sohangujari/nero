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

/** As nero/core/audit_log.py records it — one row per skill Nero ran. */
export type AuditEntry = {
  timestamp: string
  skill_name: string
  arguments: Record<string, unknown>
  result_summary: string
  provider: string
}

export type Channel = {
  enabled: boolean
  detail: string
  paired?: number
  chat_ids?: number[]
  bridge_installed?: boolean
}

export type Skill = {
  name: string
  description: string
  category: string
  tier: "read_only" | "state_changing" | "destructive"
  requires_network: boolean
  enabled: boolean
  available: boolean
}

export type Routine = {
  name: string
  schedule: string
  prompt: string
  enabled: boolean
  installed: boolean
}

export type Session = {
  session_id: string
  turns: number
  started_at: string
  last_at: string
}

export type McpServer = {
  name: string
  command: string
  args: string[]
  env_keys: string[]
  enabled: boolean
  trusted: boolean
  requires_network: boolean
}

/** Everything the sidebar's pages read, from one /api/state call. */
export type State = {
  assistant: string
  mode: string
  channels: Record<string, Channel>
  models: {
    provider: string
    model: string
    base_url: string | null
    fallback_chain: string[]
    route_by: string
    quality_rank: string[]
    health_check: boolean
    coding_model: string | null
    whitelist: string[]
    blacklist: string[]
    mode: string
    providers: string[]
    modes: string[]
    route_by_options: string[]
  }
  skills: Skill[]
  routines: Routine[]
  sessions: Session[]
  mcp: McpServer[]
  memory: {
    enabled: boolean
    facts: number
    fact_list: { key: string; value: string; source: string | null; updated_at: string }[]
    max_history_turns: number
    compact_after_messages: number
    semantic_recall: boolean
    notes_dir: string | null
    learning: boolean
    learn_after: number
    playbooks: number
    playbook_list: {
      name: string
      task: string
      steps: string
      avoid: string
      version: number
      uses: number
      updated_at: string
    }[]
  }
  counts: {
    skills_available: number
    skills_total: number
    sessions: number
    turns: number
    routines: number
    playbooks: number
    mcp: number
  }
}

export type EditAction =
  | "set"
  | "remove"
  | "forget_session"
  | "forget_fact"
  | "forget_playbook"

/** What a page calls to change something. Resolves once Nero has saved. */
export type Edit = (action: EditAction, key: string, value?: string) => Promise<void>

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

  async state(): Promise<State> {
    const response = await fetch("/api/state", { headers: this.headers() })
    if (!response.ok) throw new Error(`Nero answered with HTTP ${response.status}.`)
    return response.json()
  }

  /** One edit, answered with the state it produced — so the page renders what
   *  Nero saved rather than what it hoped would be saved. */
  async edit(action: EditAction, key: string, value?: string): Promise<State> {
    const response = await fetch("/api/edit", {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ action, key, value }),
    })
    const data = await response.json().catch(() => null)
    if (!response.ok) throw new Error(data?.error ?? `Nero refused that (HTTP ${response.status}).`)
    return data as State
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
