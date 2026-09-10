/**
 * The documentation itself: nine sections, each its own page.
 *
 * Everything enumerable is a data row rather than markup, so adding a command
 * or a skill is one line. The command text is what `nero --help` actually
 * prints; keep the two in step.
 *
 * The shell that renders these lives in ./index.tsx and is shared by the
 * dashboard and the published site, so there is one copy of the content.
 */
import { useEffect, useState, type ReactNode } from "react"
import { Check, Copy } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

/* ---------------------------------------------------------------- data --- */

type Row = { cmd: string; does: string }

const SESSIONS: Row[] = [
  { cmd: "nero", does: "The universal session: terminal chat, plus the Telegram bridge if a phone is paired." },
  { cmd: "nero chat", does: "The terminal on its own. No Telegram bridge." },
  { cmd: "nero talk", does: "Voice. Recording stops on its own; talk over Nero to interrupt its reply." },
  { cmd: "nero talk --once", does: "A single voice exchange, then exit." },
  { cmd: "nero dashboard", does: "The browser UI: chat, plus every channel, model, skill and log." },
  { cmd: "nero dashboard --port 8643", does: "Serve the dashboard on another port." },
  { cmd: "nero --version", does: "Print the installed version and exit." },
  { cmd: "nero --debug", does: "Verbose logging to stderr: tool-call plumbing and per-turn history." },
]

const CONFIG: Row[] = [
  { cmd: "nero config", does: "Interactive configuration: provider, model, keys, voice." },
  { cmd: "nero config show", does: "Print the current configuration, with API keys masked." },
  { cmd: "nero config set <key> <value>", does: "Set one value by dotted path, e.g. llm.provider gemini." },
  { cmd: "nero config set-key <provider>", does: "Store an API key in the system keyring. --slot N for key rotation." },
  { cmd: "nero config spotify", does: "Store Spotify credentials, so Nero can start a named song there." },
  { cmd: "nero detect", does: "Re-run hardware detection and refresh the local-model recommendation." },
]

const MEMORY_CMDS: Row[] = [
  { cmd: "nero facts", does: "List every fact Nero has remembered about you." },
  { cmd: "nero facts forget <key>", does: "Delete one remembered fact." },
  { cmd: "nero forget", does: "Clear remembered conversation history. The audit log is untouched." },
  { cmd: "nero history", does: "What Nero has actually done: a log of recent skill invocations. -n for more." },
  { cmd: "nero notes index", does: "Reindex your notes directory and report what changed." },
  { cmd: "nero notes search <query>", does: "Search the indexed notes. -n for more results." },
]

const TELEGRAM: Row[] = [
  { cmd: "nero telegram", does: "Run the bridge in the foreground." },
  { cmd: "nero telegram setup", does: "Store a bot token and pair the chat allowed to use it." },
  { cmd: "nero telegram install", does: "Keep the bridge running: at login, and again if it ever stops." },
  { cmd: "nero telegram uninstall", does: "Stop the background bridge and remove its launchd agent." },
  { cmd: "nero telegram pending", does: "Chats waiting to be paired. Codes are shown in Telegram, never here." },
  { cmd: "nero telegram approve <code>", does: "Pair the chat that was given this code." },
]

const AUTOMATION: Row[] = [
  { cmd: "nero routine list", does: "Configured routines: schedule, enabled, and whether installed." },
  { cmd: "nero routine run <name>", does: "One headless turn: send the routine's prompt, print the reply." },
  { cmd: "nero routine install <name>", does: "Write and load the launchd agent for a routine." },
  { cmd: "nero routine uninstall <name>", does: "Unload and remove a routine's launchd agent." },
  { cmd: "nero approvals", does: "Actions a routine queued because they need a human." },
  { cmd: "nero approvals run <id>", does: "Review one queued action and, only if you approve, run it." },
  { cmd: "nero approvals discard <id>", does: "Throw away a queued action without running it." },
  { cmd: "nero mcp", does: "List configured MCP servers and the tools they expose." },
]

type SkillRow = { name: string; group: string; on: boolean; does: string }

const SKILLS: SkillRow[] = [
  { name: "open_app", group: "Apps", on: true, does: "Open an installed application by name." },
  { name: "close_app", group: "Apps", on: true, does: "Quit a running application. Gracefully, so unsaved work still prompts." },
  { name: "open_website", group: "Apps", on: true, does: "Open a site, optionally naming which browser to use." },
  { name: "play_music", group: "Apps", on: true, does: "Start a named song, or play, pause, skip what is already going." },
  { name: "set_volume", group: "Apps", on: true, does: "Set, nudge, or mute the system output volume." },
  { name: "get_weather", group: "Web", on: true, does: "Current weather and a short forecast for a place." },
  { name: "web_search", group: "Web", on: true, does: "Search the web for titles, URLs and snippets." },
  { name: "fetch_web_page", group: "Web", on: true, does: "Fetch one page and return its text, markup stripped." },
  { name: "read_file", group: "Files", on: true, does: "Read a text file. Refuses binaries and oversized files." },
  { name: "write_file", group: "Files", on: false, does: "Write a file, creating parent directories as needed." },
  { name: "edit_file", group: "Files", on: false, does: "Replace an exact, unique snippet inside a file." },
  { name: "move_path", group: "Files", on: false, does: "Move or rename. Refuses to overwrite anything." },
  { name: "delete_path", group: "Files", on: false, does: "Delete a file or directory." },
  { name: "remember_fact", group: "Memory", on: true, does: "Store one fact about you under an explicit key." },
  { name: "recall_facts", group: "Memory", on: true, does: "Look up facts remembered earlier." },
  { name: "forget_fact", group: "Memory", on: false, does: "Delete one remembered fact by key." },
  { name: "search_notes", group: "Memory", on: true, does: "Search your own notes directory." },
  { name: "run_shell", group: "Code", on: false, does: "Run a shell command in the working directory." },
  { name: "run_git", group: "Code", on: false, does: "Run git. push is refused unless allowlisted." },
  { name: "run_python", group: "Code", on: false, does: "Run a Python snippet in a fresh subprocess." },
  { name: "run_javascript", group: "Code", on: false, does: "Run a JavaScript snippet with Node.js." },
]

const SETTINGS: Row[] = [
  { cmd: "llm.provider", does: "claude, openai, gemini, ollama, bedrock, and more." },
  { cmd: "llm.model", does: "The model id for that provider." },
  { cmd: "llm.fallback_chain", does: "Providers to try, in order, when the primary fails." },
  { cmd: "mode", does: "online or offline. Offline keeps everything on your machine." },
  { cmd: "assistant.name", does: "What Nero calls itself." },
  { cmd: "skills.enabled.<skill>", does: "Turn one skill on or off." },
  { cmd: "skills.weather.default_location", does: "Where 'what's the weather' means, with no place named." },
  { cmd: "skills.music.preferred_app", does: "Spotify or Music. Asked once, then remembered." },
  { cmd: "skills.browser.preferred", does: "Which browser 'open YouTube' should use." },
  { cmd: "memory.max_history_turns", does: "How many recent turns stay in the live window." },
  { cmd: "memory.semantic_recall", does: "Vector recall alongside keyword search. Needs a local embedder." },
  { cmd: "memory.notes_dir", does: "The directory search_notes reads." },
  { cmd: "voice.stt.language", does: "Speech-recognition language. en by default." },
  { cmd: "voice.tts.voice_id", does: "Which synthesized voice speaks." },
  { cmd: "security.command_denylist", does: "Substrings run_shell will never execute." },
  { cmd: "security.max_cost_usd_per_session", does: "Hard ceiling on spend. 0 means no ceiling." },
]

/* ---------------------------------------------------------- primitives --- */

export function Cmd({ children }: { children: string }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1400)
    return () => clearTimeout(timer)
  }, [copied])

  return (
    <div className="bg-muted/50 group relative my-4 rounded-lg border">
      <pre className="overflow-x-auto px-4 py-3 pr-12 font-mono text-[13px] leading-relaxed">
        <code>{children}</code>
      </pre>
      <button
        type="button"
        aria-label="Copy"
        onClick={() => {
          void navigator.clipboard?.writeText(children).then(() => setCopied(true))
        }}
        className="text-muted-foreground hover:bg-background hover:text-foreground absolute top-2 right-2 rounded-md border border-transparent p-1.5 opacity-0 transition group-hover:opacity-100 focus-visible:opacity-100"
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </button>
    </div>
  )
}

/** Inline code. Short name because the prose is dense with it. */
export function K({ children }: { children: ReactNode }) {
  return (
    <code className="bg-muted rounded px-[0.35rem] py-[0.15rem] font-mono text-[0.85em]">
      {children}
    </code>
  )
}

/** A boxed aside. `tone` picks the accent: neutral for a footnote, "note" for
 *  something the reader should act on before moving past it. */
export function Callout({
  tone = "muted",
  children,
}: {
  tone?: "muted" | "note"
  children: ReactNode
}) {
  return (
    <div
      className={
        tone === "note"
          ? "my-5 rounded-lg border border-emerald-600/40 bg-emerald-600/10 px-4 py-3 text-sm leading-relaxed"
          : "text-muted-foreground my-5 rounded-lg border px-4 py-3 text-sm leading-relaxed"
      }
    >
      {children}
    </div>
  )
}

export function Cards({ items }: { items: { title: string; body: string }[] }) {
  return (
    <div className="my-5 grid gap-3 sm:grid-cols-3">
      {items.map((item) => (
        <div key={item.title} className="rounded-lg border p-4">
          <p className="font-medium">{item.title}</p>
          <p className="text-muted-foreground mt-1.5 text-sm leading-relaxed">{item.body}</p>
        </div>
      ))}
    </div>
  )
}

/** A section's sub-heading. Its id is what the "On this page" rail links to,
 *  so it must match the `headings` entry declared alongside the body. */
export function H3({ id, children }: { id: string; children: ReactNode }) {
  return (
    <h3 id={id} className="scroll-mt-24 pt-8 text-lg font-semibold tracking-tight">
      {children}
    </h3>
  )
}

function Commands({ rows, head = "Command" }: { rows: Row[]; head?: string }) {
  return (
    <Table className="my-4">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[19rem]">{head}</TableHead>
          <TableHead>What it does</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => (
          <TableRow key={row.cmd}>
            <TableCell className="align-top font-mono text-[13px] [overflow-wrap:anywhere]">
              {row.cmd}
            </TableCell>
            <TableCell className="text-muted-foreground align-top">{row.does}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

/* ------------------------------------------------------------ sections --- */

export type Heading = { id: string; label: string }
export type Section = {
  id: string
  title: string
  lede: string
  group: string
  headings: Heading[]
  body: ReactNode
}

export const SECTIONS: Section[] = [
  {
    id: "introduction",
    title: "Introduction",
    lede: "What Nero is, and what makes it more than a chat window.",
    group: "Getting started",
    headings: [
      { id: "one-assistant", label: "One assistant, four ways" },
      { id: "any-model", label: "Any model" },
      { id: "doing-things", label: "Doing things" },
    ],
    body: (
      <>
        <Callout tone="note">
          <strong className="font-medium">New here?</strong> Read this page, then Install and
          Setup. Fifteen minutes and Nero is answering in your terminal and on your phone.
        </Callout>

        <H3 id="one-assistant">One assistant, four ways</H3>
        <p>
          Nero is reachable from the terminal, your voice, Telegram, and a browser dashboard. Every
          route runs the same turn, so a question you ask on your phone sees the same memory, the
          same skills, and the same model as one you type at the command line.
        </p>
        <Cards
          items={[
            { title: "Terminal", body: "Streaming conversation where you already work." },
            { title: "Voice", body: "Speak, and talk over the reply to interrupt it." },
            { title: "Telegram", body: "The same assistant, from your phone." },
          ]}
        />

        <H3 id="any-model">Any model</H3>
        <p>
          Nero talks to Claude, GPT, Gemini, Bedrock, or a local Ollama model, and switching between
          them is a config change rather than a code change. Set <K>mode</K> to <K>offline</K> and
          nothing leaves your machine.
        </p>
        <p>
          A fallback chain covers the case where your primary provider is down or rate limited: Nero
          tries the next entry rather than failing the turn.
        </p>

        <H3 id="doing-things">Doing things</H3>
        <p>
          What separates Nero from a chat window is that the model can call skills: open an app,
          play a song, read a file, search the web. Anything destructive is off by default, and
          everything that runs is written to an audit log.
        </p>
        <Cmd>nero history</Cmd>
      </>
    ),
  },
  {
    id: "install",
    title: "Install",
    lede: "How to get Nero onto your machine.",
    group: "Getting started",
    headings: [
      { id: "binary", label: "Standalone binary" },
      { id: "source", label: "From source" },
    ],
    body: (
      <>
        <H3 id="binary">Standalone binary</H3>
        <p>
          Nero ships as a standalone binary with a fixed Python bundled in. There is nothing to
          install alongside it. Download the asset for your OS from the releases page, make it
          executable, and run it.
        </p>
        <Cmd>{`chmod +x nero-macos-arm64
./nero-macos-arm64 --version`}</Cmd>

        <H3 id="source">From source</H3>
        <p>To work on Nero itself, clone the repo and let uv resolve the environment.</p>
        <Cmd>{`git clone https://github.com/sohangujari/nero
cd nero
uv sync --extra voice`}</Cmd>
        <Callout>
          The <K>voice</K> extra pulls in speech recognition and synthesis. Skip it and everything
          except <K>nero talk</K> still works.
        </Callout>
      </>
    ),
  },
  {
    id: "setup",
    title: "Setup",
    lede: "Point Nero at a model and give it a key.",
    group: "Getting started",
    headings: [
      { id: "cloud", label: "A cloud provider" },
      { id: "local", label: "A local model" },
      { id: "keys", label: "Where keys live" },
    ],
    body: (
      <>
        <p>
          First run detects your hardware and recommends a local model. After that, point Nero at
          whichever provider you want.
        </p>

        <H3 id="cloud">A cloud provider</H3>
        <Cmd>{`nero config set llm.provider gemini
nero config set llm.model gemini-3.5-flash-lite
nero config set-key gemini`}</Cmd>
        <p>
          Running <K>nero config</K> with no arguments walks the same choices interactively.
        </p>

        <H3 id="local">A local model</H3>
        <p>
          Choose <K>ollama</K> for a fully local setup. Nero checks the server is up and offers to
          pull the model if it is missing. No key is asked for.
        </p>
        <Cmd>{`nero config set llm.provider ollama
nero config set llm.model qwen3`}</Cmd>
        <Callout tone="note">
          Not every small local model can handle skills. Some call one on every message, including
          plain conversation. See Troubleshooting before you settle on one.
        </Callout>

        <H3 id="keys">Where keys live</H3>
        <p>
          Keys go into your system keyring, never into the config file. <K>nero config show</K>{" "}
          masks them. Pass <K>--slot N</K> to store more than one key per provider, which Nero
          rotates through when it hits a rate limit.
        </p>
      </>
    ),
  },
  {
    id: "talking",
    title: "Talking to Nero",
    lede: "The terminal, your voice, and your phone.",
    group: "Using Nero",
    headings: [
      { id: "sessions", label: "Sessions" },
      { id: "voice", label: "Voice" },
      { id: "phone", label: "Your phone" },
    ],
    body: (
      <>
        <H3 id="sessions">Sessions</H3>
        <p>
          <K>nero</K> on its own is the universal session: the terminal conversation, plus the
          Telegram bridge if you have paired a phone. The other entry points are that same session
          narrowed to one channel.
        </p>
        <Commands rows={SESSIONS} />

        <H3 id="voice">Voice</H3>
        <p>
          In <K>nero talk</K>, recording stops on its own when you stop speaking, and you can talk
          over a reply to interrupt it. A waveform shows the level it is hearing.
        </p>

        <H3 id="phone">Your phone</H3>
        <p>
          Create a bot with Telegram's BotFather, then hand Nero the token. Only chats you pair can
          talk to it.
        </p>
        <Cmd>{`nero telegram setup
nero telegram install`}</Cmd>
      </>
    ),
  },
  {
    id: "commands",
    title: "Commands",
    lede: "Every command, and what it does.",
    group: "Using Nero",
    headings: [
      { id: "cmd-config", label: "Configuration" },
      { id: "cmd-memory", label: "Memory and history" },
      { id: "cmd-telegram", label: "Telegram" },
      { id: "cmd-automation", label: "Routines and MCP" },
    ],
    body: (
      <>
        <Callout>
          Every command takes <K>--help</K>, and that output is the authority. This page is a map,
          not a replacement.
        </Callout>

        <H3 id="cmd-config">Configuration</H3>
        <Commands rows={CONFIG} />

        <H3 id="cmd-memory">Memory and history</H3>
        <Commands rows={MEMORY_CMDS} />

        <H3 id="cmd-telegram">Telegram</H3>
        <Commands rows={TELEGRAM} />

        <H3 id="cmd-automation">Routines, approvals and MCP</H3>
        <Commands rows={AUTOMATION} />
      </>
    ),
  },
  {
    id: "skills",
    title: "Skills",
    lede: "What Nero can actually do on your machine.",
    group: "Using Nero",
    headings: [
      { id: "skill-list", label: "The built-in skills" },
      { id: "skill-toggle", label: "Turning them on and off" },
    ],
    body: (
      <>
        <p>
          A skill is something Nero can actually do, offered to the model as a tool. Anything that
          writes, deletes, or executes is off until you turn it on.
        </p>

        <H3 id="skill-list">The built-in skills</H3>
        <Table className="my-4">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[11rem]">Skill</TableHead>
              <TableHead className="w-[6rem]">Group</TableHead>
              <TableHead className="w-[6rem]">Default</TableHead>
              <TableHead>What it does</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {SKILLS.map((skill) => (
              <TableRow key={skill.name}>
                <TableCell className="align-top font-mono text-[13px] [overflow-wrap:anywhere]">
                  {skill.name}
                </TableCell>
                <TableCell className="text-muted-foreground align-top">{skill.group}</TableCell>
                <TableCell className="align-top">
                  <Badge variant={skill.on ? "secondary" : "outline"}>
                    {skill.on ? "on" : "off"}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground align-top">{skill.does}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <H3 id="skill-toggle">Turning them on and off</H3>
        <Cmd>nero config set skills.enabled.run_python true</Cmd>
        <p>
          MCP servers add their tools to this same list. <K>nero mcp</K> shows what each one
          exposes.
        </p>
      </>
    ),
  },
  {
    id: "configuration",
    title: "Configuration",
    lede: "One YAML file, edited by dotted path.",
    group: "Reference",
    headings: [{ id: "settings", label: "Settings" }],
    body: (
      <>
        <p>
          Configuration is one YAML file, edited with <K>nero config set</K>. Unknown keys are
          rejected rather than silently ignored, so a typo tells you.
        </p>

        <H3 id="settings">Settings</H3>
        <Commands rows={SETTINGS} head="Setting" />
        <p>
          <K>nero config show</K> prints the whole file. The dashboard's Config page is the same
          thing, editable in place.
        </p>
      </>
    ),
  },
  {
    id: "memory",
    title: "Memory",
    lede: "Three separate things, cleared separately.",
    group: "Reference",
    headings: [
      { id: "facts", label: "Facts" },
      { id: "history", label: "Conversation history" },
      { id: "audit", label: "The audit log" },
    ],
    body: (
      <>
        <H3 id="facts">Facts</H3>
        <p>
          Facts are explicit key-value details Nero was told to keep, like a default city or a
          preferred music player. They go into every prompt, so a wrong one is worth deleting.
        </p>
        <Cmd>{`nero facts
nero facts forget favorite_color`}</Cmd>

        <H3 id="history">Conversation history</H3>
        <p>
          Recent turns stay in a live window. Older ones are retrieved only when they are relevant
          to the message you just sent, by keyword search fused with vector search, which keeps the
          prompt small without forgetting anything.
        </p>
        <Cmd>nero forget</Cmd>

        <H3 id="audit">The audit log</H3>
        <p>
          The audit log records every skill Nero ran and what came back. <K>nero forget</K>{" "}
          deliberately leaves it alone: it is the record of what actually happened, and the only
          way to tell a real action from a narrated one.
        </p>
        <Cmd>nero history -n 50</Cmd>
      </>
    ),
  },
  {
    id: "troubleshooting",
    title: "Troubleshooting",
    lede: "When Nero is slow, wrong, or refusing.",
    group: "Reference",
    headings: [
      { id: "slow", label: "Slow or wrong replies" },
      { id: "refusing", label: "Refusing something it can do" },
      { id: "claiming", label: "Claiming it did something" },
      { id: "debug", label: "Anything else" },
    ],
    body: (
      <>
        <H3 id="slow">Slow or wrong replies</H3>
        <p>
          Small local models often cannot tell when <em>not</em> to call a skill. A model in that
          state answers "hi" by running a tool, which is both wrong and several seconds slower,
          because the skill schemas dominate the prompt and a tool call costs a second round trip.
          Nero warns at startup for models measured to do this.
        </p>
        <p>Move to a larger local model, or to a cloud provider.</p>
        <Cmd>nero config set llm.provider gemini</Cmd>

        <H3 id="refusing">Refusing something it can do</H3>
        <p>
          The skill is probably disabled. Check the Skills page, and confirm with <K>nero history</K>{" "}
          whether it was refused or never attempted.
        </p>

        <H3 id="claiming">Claiming it did something</H3>
        <p>
          The audit log is ground truth. If <K>nero history</K> shows no entry, the model narrated
          an action instead of taking one.
        </p>

        <H3 id="debug">Anything else</H3>
        <p>Re-run with tool-call plumbing and per-turn history on stderr.</p>
        <Cmd>nero --debug chat</Cmd>
      </>
    ),
  },
]

export const GROUPS = ["Getting started", "Using Nero", "Reference"] as const
