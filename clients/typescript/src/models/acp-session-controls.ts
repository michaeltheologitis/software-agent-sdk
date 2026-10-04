// Hand-written until the pinned agent-server release carries these schemas;
// field for field with openhands.sdk.agent.acp_models.

export type ACPConfigOptionType = 'select' | 'boolean';

/** The text a command takes after its name. */
export interface ACPCommandInput {
  hint: string;
}

/** One slash command an ACP session offers; invoked as a message starting `/<name>`. */
export interface ACPAvailableCommand {
  name: string;
  description: string;
  input?: ACPCommandInput | null;
}

/** One value of a select option. */
export interface ACPConfigOptionValue {
  value: string;
  name: string;
  description?: string | null;
  group?: string | null;
}

/** One session config option; select groups are flattened into `options`. */
export interface ACPConfigOption {
  id: string;
  name: string;
  type: ACPConfigOptionType;
  current_value: string | boolean;
  description?: string | null;
  category?: string | null;
  options: ACPConfigOptionValue[];
}

/** The slash commands and config options an ACP session offers now. */
export interface ACPSessionControls {
  available_commands: ACPAvailableCommand[];
  config_options: ACPConfigOption[];
}

/** Option values to apply at a conversation's start: `{ [config_id]: value }`. */
export type ACPConfigOptionValues = Record<string, string | boolean>;

export interface ACPConfigOptionSetRequest {
  config_id: string;
  value: string | boolean;
}

export interface ACPConfigOptionSetResponse {
  /** True when a live session took the value; false when it waits for the start. */
  applied: boolean;
  /** The session's controls after the set; empty when not applied. */
  controls: ACPSessionControls;
}

/**
 * The `kind` the events search matches for an ACPSessionControlsEvent: the
 * search filters by the event's module-qualified class name.
 */
export const ACP_SESSION_CONTROLS_EVENT_KIND =
  'openhands.sdk.event.acp_session_controls.ACPSessionControlsEvent';
