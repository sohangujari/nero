import type { ReactNode } from "react"

import { Remove, SelectField, TextField, Toggle } from "@/components/editable"
import { Empty, Field, Page } from "@/components/page"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { Edit, State } from "@/lib/api"

type PageProps = { state: State; edit: Edit }

const KEYRING_NOTE =
  "API keys are the exception: they live in the OS keyring and are neither shown nor set here. Use `nero config set-key`."

/** A label/control row — the editable twin of `Field`. */
function Setting({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="flex items-center gap-4 border-b py-3 last:border-0">
      <div className="w-56 shrink-0">
        <p className="text-sm">{label}</p>
        {hint && <p className="text-muted-foreground text-xs">{hint}</p>}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  )
}

// --- Channels ---------------------------------------------------------------

const CHANNEL_COPY: Record<string, string> = {
  terminal: "The session you started this from.",
  dashboard: "This page. Token-gated, loopback only.",
  voice: "Hands-free: speech in, speech out.",
  telegram: "Your phone. Only paired chats are answered.",
}

/** Which config key switches a channel on, where one exists. The terminal and
 *  this page are not switchable — you are already using them. */
const CHANNEL_KEY: Record<string, string> = {
  voice: "voice.enabled",
  telegram: "telegram.enabled",
}

export function Channels({ state, edit }: PageProps) {
  const paired = state.channels.telegram?.chat_ids ?? []
  return (
    <Page title="Channels" lede="Every way a person can reach Nero. Switch one off and it stops answering.">
      <div className="grid gap-4 sm:grid-cols-2">
        {Object.entries(state.channels).map(([name, channel]) => (
          <Card key={name}>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <CardTitle className="text-base capitalize">{name}</CardTitle>
                {CHANNEL_KEY[name] ? (
                  <div className="ml-auto">
                    <Toggle
                      field={CHANNEL_KEY[name]}
                      on={channel.enabled}
                      edit={edit}
                      label={`${name} enabled`}
                    />
                  </div>
                ) : (
                  <Badge variant="secondary" className="ml-auto">
                    always on
                  </Badge>
                )}
              </div>
              <CardDescription>{CHANNEL_COPY[name] ?? ""}</CardDescription>
            </CardHeader>
            <CardContent className="text-muted-foreground text-sm">
              {channel.detail}
              {name === "telegram" && (
                <div className="mt-3">
                  {paired.length === 0 ? (
                    <p className="text-foreground text-xs">
                      No chat paired yet. Message the bot, then run{" "}
                      <code>nero telegram approve &lt;code&gt;</code>.
                    </p>
                  ) : (
                    <ul className="space-y-1">
                      {paired.map((id) => (
                        <li key={id} className="flex items-center gap-2">
                          <span className="font-mono text-xs">{id}</span>
                          <Remove
                            what={`chat ${id}`}
                            detail="That phone stops being answered. It can pair again with a new code."
                            onConfirm={() =>
                              edit(
                                "set",
                                "telegram.allowed_chat_ids",
                                paired.filter((other) => other !== id).join(","),
                              )
                            }
                          />
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </Page>
  )
}

// --- Models -----------------------------------------------------------------

export function Models({ state, edit }: PageProps) {
  const m = state.models
  return (
    <Page
      title="Models"
      lede={`What answers now, and what answers if that fails. Saved the moment you change it — same validation as \`nero config set\`. ${KEYRING_NOTE}`}
    >
      <Setting label="Provider">
        <SelectField field="llm.provider" value={m.provider} options={m.providers} edit={edit} />
      </Setting>
      <Setting label="Model" hint="Exact model id">
        <TextField field="llm.model" value={m.model} edit={edit} />
      </Setting>
      <Setting label="Mode" hint="Offline withdraws every network skill">
        <SelectField field="mode" value={m.mode} options={m.modes} edit={edit} />
      </Setting>
      <Setting label="Base URL" hint="Custom endpoints only">
        <TextField field="llm.base_url" value={m.base_url ?? ""} edit={edit} />
      </Setting>
      <Setting label="Fallback chain" hint="provider/model, comma separated">
        <TextField
          field="llm.fallback_chain"
          value={m.fallback_chain.join(", ")}
          edit={edit}
          className="h-8 max-w-lg font-mono text-xs"
        />
      </Setting>
      <Setting label="Route by">
        <SelectField
          field="llm.route_by"
          value={m.route_by}
          options={m.route_by_options}
          edit={edit}
        />
      </Setting>
      <Setting label="Coding model" hint="Used for /code">
        <TextField field="llm.coding_model" value={m.coding_model ?? ""} edit={edit} />
      </Setting>
      <Setting label="Whitelist" hint="Comma separated; empty means all">
        <TextField field="llm.model_whitelist" value={m.whitelist.join(", ")} edit={edit} />
      </Setting>
      <Setting label="Blacklist" hint="Comma separated">
        <TextField field="llm.model_blacklist" value={m.blacklist.join(", ")} edit={edit} />
      </Setting>
      <Setting label="Health check" hint="Skip a chain entry after two failures">
        <Toggle field="llm.health_check" on={m.health_check} edit={edit} label="health check" />
      </Setting>
    </Page>
  )
}

// --- Skills -----------------------------------------------------------------

const TIER: Record<string, "secondary" | "outline" | "destructive"> = {
  read_only: "outline",
  state_changing: "secondary",
  destructive: "destructive",
}

export function Skills({ state, edit }: PageProps) {
  if (state.skills.length === 0) {
    return (
      <Page title="Skills" lede="What the model may call on this machine.">
        <Empty>No skill registry attached to this session.</Empty>
      </Page>
    )
  }
  return (
    <Page
      title="Skills"
      lede="What the model may call on this machine. The switch is the setting; `available` is the outcome — a network skill stays withdrawn in offline mode however you switch it."
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Skill</TableHead>
            <TableHead>Permission</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Enabled</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {state.skills.map((skill) => (
            <TableRow key={skill.name}>
              <TableCell className={skill.available ? "" : "opacity-60"}>
                <span className="font-mono text-xs font-medium">{skill.name}</span>
                <p className="text-muted-foreground mt-0.5 text-xs">{skill.description}</p>
              </TableCell>
              <TableCell className="align-top">
                <Badge variant={TIER[skill.tier] ?? "outline"}>{skill.tier.replace("_", " ")}</Badge>
                {skill.requires_network && (
                  <Badge variant="outline" className="ml-1">
                    network
                  </Badge>
                )}
              </TableCell>
              <TableCell className="align-top">
                {skill.available ? (
                  <Badge variant="secondary">available</Badge>
                ) : (
                  <Badge variant="outline">{skill.enabled ? "blocked by mode" : "off"}</Badge>
                )}
              </TableCell>
              <TableCell className="text-right align-top">
                <Toggle
                  field={`skills.enabled.${skill.name}`}
                  on={skill.enabled}
                  edit={edit}
                  label={skill.name}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="text-muted-foreground mt-4 text-xs">
        Destructive skills stay gated behind a confirmation at the point of use, however they are
        switched here.
      </p>
    </Page>
  )
}

// --- Routines ---------------------------------------------------------------

export function Routines({ state, edit }: PageProps) {
  return (
    <Page
      title="Routines"
      lede="Prompts Nero runs on a schedule. `Installed` is separate from `enabled` on purpose: a routine written to config but never installed will never fire — switching it on here does not install it, `nero routine install` does."
    >
      {state.routines.length === 0 ? (
        <Empty>
          Nothing scheduled. Add one with <code>nero routine add</code>.
        </Empty>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Schedule</TableHead>
              <TableHead>Prompt</TableHead>
              <TableHead>launchd</TableHead>
              <TableHead className="text-right">Enabled</TableHead>
              <TableHead className="w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {state.routines.map((routine) => (
              <TableRow key={routine.name}>
                <TableCell className="font-mono text-xs font-medium">{routine.name}</TableCell>
                <TableCell>
                  <TextField
                    field={`routines.routines.${routine.name}.schedule`}
                    value={routine.schedule}
                    edit={edit}
                    className="h-8 w-36 font-mono text-xs"
                  />
                </TableCell>
                <TableCell>
                  <TextField
                    field={`routines.routines.${routine.name}.prompt`}
                    value={routine.prompt}
                    edit={edit}
                    className="h-8 w-full min-w-48 text-xs"
                  />
                </TableCell>
                <TableCell>
                  <Badge variant={routine.installed ? "secondary" : "outline"}>
                    {routine.installed ? "installed" : "not installed"}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  <Toggle
                    field={`routines.routines.${routine.name}.enabled`}
                    on={routine.enabled}
                    edit={edit}
                    label={routine.name}
                  />
                </TableCell>
                <TableCell>
                  <Remove
                    what={`routine "${routine.name}"`}
                    detail={
                      routine.installed
                        ? "This removes it from config. It is still loaded in launchd until you run `nero routine uninstall`."
                        : "This removes it from config. Nothing else is touched."
                    }
                    onConfirm={() => edit("remove", `routines.routines.${routine.name}`)}
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

// --- MCP --------------------------------------------------------------------

export function Mcp({ state, edit }: PageProps) {
  return (
    <Page
      title="MCP servers"
      lede="External tool servers Nero starts. Their environment is shown by key name only — the values are commonly secrets, so they are neither displayed nor editable here."
    >
      {state.mcp.length === 0 ? (
        <Empty>
          None configured. Add one with <code>nero mcp add</code>.
        </Empty>
      ) : (
        <div className="grid gap-4">
          {state.mcp.map((server) => (
            <Card key={server.name}>
              <CardHeader className="pb-3">
                <div className="flex flex-wrap items-center gap-2">
                  <CardTitle className="text-base">{server.name}</CardTitle>
                  {server.requires_network && <Badge variant="outline">network</Badge>}
                  <div className="ml-auto flex items-center gap-2">
                    <Toggle
                      field={`mcp.servers.${server.name}.enabled`}
                      on={server.enabled}
                      edit={edit}
                      label={`${server.name} enabled`}
                    />
                    <Remove
                      what={`MCP server "${server.name}"`}
                      detail="Nero stops starting it. Its own files and installation are untouched."
                      onConfirm={() => edit("remove", `mcp.servers.${server.name}`)}
                    />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Setting label="Trusted" hint="Skips the per-call confirmation">
                  <Toggle
                    field={`mcp.servers.${server.name}.trusted`}
                    on={server.trusted}
                    edit={edit}
                    label={`${server.name} trusted`}
                  />
                </Setting>
                <Setting label="Command">
                  <TextField
                    field={`mcp.servers.${server.name}.command`}
                    value={server.command}
                    edit={edit}
                  />
                </Setting>
                <dl>
                  <Field label="Arguments">{server.args.join(" ") || "—"}</Field>
                  <Field label="Environment">{server.env_keys.join(", ") || "—"}</Field>
                </dl>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </Page>
  )
}

// --- Memory -----------------------------------------------------------------

export function Memory({ state, edit }: PageProps) {
  const m = state.memory
  return (
    <Page
      title="Memory"
      lede="Nero keeps a short window of the current conversation and searches the rest on demand, so a long history costs recall rather than every prompt."
    >
      <Setting label="Memory" hint="Off means every turn starts cold">
        <Toggle field="memory.enabled" on={m.enabled} edit={edit} label="memory enabled" />
      </Setting>
      <Setting label="Semantic recall" hint="Adds embedding search to keyword search">
        <Toggle
          field="memory.semantic_recall"
          on={m.semantic_recall}
          edit={edit}
          label="semantic recall"
        />
      </Setting>
      <Setting label="Turns kept in the prompt" hint="Higher is slower and costs more">
        <TextField
          field="memory.max_history_turns"
          value={String(m.max_history_turns)}
          edit={edit}
          className="h-8 w-24 font-mono text-xs"
        />
      </Setting>
      <Setting label="Compact after" hint="Messages before the window is trimmed">
        <TextField
          field="memory.compact_after_messages"
          value={String(m.compact_after_messages)}
          edit={edit}
          className="h-8 w-24 font-mono text-xs"
        />
      </Setting>
      <Setting label="Notes directory" hint="Indexed for search; blank to disable">
        <TextField
          field="memory.notes_dir"
          value={m.notes_dir ?? ""}
          edit={edit}
          className="h-8 max-w-lg font-mono text-xs"
        />
      </Setting>

      <h3 className="mt-8 mb-1 text-sm font-semibold">Remembered facts</h3>
      <p className="text-muted-foreground mb-3 text-xs">
        Things Nero was told to keep. Removing one is immediate and cannot be undone from here.
      </p>
      {m.fact_list.length === 0 ? (
        <Empty>Nothing remembered yet.</Empty>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Key</TableHead>
              <TableHead>Value</TableHead>
              <TableHead className="w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {m.fact_list.map((fact) => (
              <TableRow key={fact.key}>
                <TableCell className="font-mono text-xs font-medium">{fact.key}</TableCell>
                <TableCell className="text-sm [overflow-wrap:anywhere]">{fact.value}</TableCell>
                <TableCell>
                  <Remove
                    what={`"${fact.key}"`}
                    detail="Nero forgets this immediately, in every channel."
                    onConfirm={() => edit("forget_fact", fact.key)}
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
