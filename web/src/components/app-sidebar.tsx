import {
  BookOpen,
  Blocks,
  CalendarClock,
  Cpu,
  Database,
  LayoutDashboard,
  MessagesSquare,
  Plug,
  Radio,
  ScrollText,
  SlidersHorizontal,
  History,
} from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"

/** The page keys the sidebar can select. Dashboard is first, and is the
 *  landing page — the sidebar's order is the order of the reading. */
export const PAGES = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, group: "" },
  { id: "chat", label: "Chat", icon: MessagesSquare, group: "" },
  { id: "docs", label: "Docs", icon: BookOpen, group: "" },
  { id: "channels", label: "Channels", icon: Radio, group: "Interfaces" },
  { id: "sessions", label: "Sessions", icon: History, group: "Interfaces" },
  { id: "logs", label: "Logs", icon: ScrollText, group: "Interfaces" },
  { id: "models", label: "Models", icon: Cpu, group: "Setup" },
  { id: "skills", label: "Skills", icon: Blocks, group: "Setup" },
  { id: "routines", label: "Routines", icon: CalendarClock, group: "Setup" },
  { id: "mcp", label: "MCP servers", icon: Plug, group: "Setup" },
  { id: "memory", label: "Memory", icon: Database, group: "Setup" },
  { id: "config", label: "Config", icon: SlidersHorizontal, group: "Setup" },
] as const

export type PageId = (typeof PAGES)[number]["id"]

const GROUPS = ["", "Interfaces", "Setup"] as const

export function AppSidebar({
  name,
  mode,
  page,
  onSelect,
  counts,
}: {
  name: string
  mode: string
  page: PageId
  onSelect: (id: PageId) => void
  counts: Partial<Record<PageId, number>>
}) {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" className="cursor-default hover:bg-transparent">
              <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg text-sm font-semibold">
                {name.slice(0, 1).toUpperCase()}
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">{name}</span>
                <span className="text-muted-foreground truncate text-xs">{mode} · local</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {GROUPS.map((group) => (
          <SidebarGroup key={group || "main"}>
            {group && <SidebarGroupLabel>{group}</SidebarGroupLabel>}
            <SidebarGroupContent>
              <SidebarMenu>
                {PAGES.filter((p) => p.group === group).map((p) => (
                  <SidebarMenuItem key={p.id}>
                    <SidebarMenuButton
                      isActive={page === p.id}
                      tooltip={p.label}
                      onClick={() => onSelect(p.id)}
                    >
                      <p.icon />
                      <span>{p.label}</span>
                    </SidebarMenuButton>
                    {counts[p.id] ? <SidebarMenuBadge>{counts[p.id]}</SidebarMenuBadge> : null}
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <p className="text-muted-foreground px-2 pb-1 text-[11px] leading-snug group-data-[collapsible=icon]:hidden">
          Read-only except Chat. Change anything with <code>nero config</code>.
        </p>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
