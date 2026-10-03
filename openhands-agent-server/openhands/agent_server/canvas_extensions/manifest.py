"""Canvas Extensions manifest: schema, validation, and entrypoint containment.

A Canvas extension is an installable UI bundle that contributes pages to the
OpenHands Canvas frontend. Extensions are installed and served entirely by
the agent-server (via ``openhands.sdk.extensions.installation``, the same
type-agnostic install-tracking framework Plugins/Skills use); nothing here
is consumed by ``Agent``/``Conversation``.

This module defines the manifest schema (``canvas-extension.json``) and the
two security-critical checks around it:

* Name / contribution-id / page-path validation (syntactic, on the model).
* Entrypoint containment (filesystem-level, once a package root is known) —
  rejects both textual path traversal and symlink escapes.
"""

import re
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from openhands.sdk.extensions.installation.utils import validate_extension_name


# Filename a canvas extension's manifest is loaded from, at its package root.
MANIFEST_FILENAME: Final[str] = "canvas-extension.json"

# Absolute, kebab-case, multi-segment UI route, e.g. "/dashboard/settings".
_PAGE_PATH_PATTERN: re.Pattern[str] = re.compile(
    r"^/[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)*$"
)
# "/" or an absolute kebab-case path; a tab's place inside its panel.
_TAB_PATH_PATTERN: re.Pattern[str] = re.compile(
    r"^/(?:[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)*)?$"
)
PANEL_ICON_MEDIA_TYPES: Final[dict[str, str]] = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def _validate_contribution_id(value: str) -> str:
    """Refuse an id that is not kebab-case, as validate_extension_name does."""
    try:
        validate_extension_name(value)
    except ValueError as e:
        raise ValueError(
            f"Invalid contribution id. Expected kebab-case, got {value!r}."
        ) from e
    return value


ContributionId = Annotated[str, AfterValidator(_validate_contribution_id)]


class CanvasExtensionPage(BaseModel):
    """A single page contributed to the Canvas UI by an extension."""

    id: ContributionId = Field(
        description="Unique contribution id within the extension"
    )
    title: str = Field(description="Page title shown in Canvas navigation")
    path: str = Field(description="Route the page is mounted at, e.g. '/dashboard'")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not _PAGE_PATH_PATTERN.fullmatch(v):
            raise ValueError(
                "Invalid page path. Expected an absolute kebab-case route "
                f"(e.g. '/dashboard'), got {v!r}."
            )
        return v


class CanvasExtensionPanelTab(BaseModel):
    """One tab of a conversation panel; its page mounts when the tab is selected."""

    id: ContributionId = Field(
        description="Contribution id; the id the App registers this tab's page under",
    )
    title: str = Field(min_length=1, description="Tab label in the panel's tab row")
    path: str = Field(
        default="/",
        description="Where the tab's page starts inside the panel; '/' is its root",
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not _TAB_PATH_PATTERN.fullmatch(value):
            raise ValueError(
                "Invalid tab path. Expected '/' or an absolute kebab-case path "
                f"(e.g. '/create'), got {value!r}."
            )
        return value


class CanvasExtensionConversationPanel(BaseModel):
    """A panel opened from a button in the conversation header."""

    id: ContributionId = Field(description="Contribution id of the panel")
    title: str = Field(
        min_length=1,
        description="Panel title; the header button's tooltip is 'Show' and this",
    )
    icon: str | None = Field(
        default=None, description="Package-relative .svg or .png for the header button"
    )
    tabs: list[CanvasExtensionPanelTab] = Field(
        min_length=1, description="The panel's tabs, in tab-row order"
    )

    @field_validator("icon")
    @classmethod
    def _validate_icon(cls, value: str | None) -> str | None:
        """Reject textual traversal, absolute paths and other file types.

        Syntactic only -- see :func:`resolve_panel_icon` for the symlink-aware
        containment check against the installed package root.
        """
        if value is None:
            return value
        if value.startswith("/") or ".." in Path(value).parts:
            raise ValueError("panel icon must be a package-relative path")
        if Path(value).suffix not in PANEL_ICON_MEDIA_TYPES:
            raise ValueError("panel icon must be a .svg or .png file")
        return value

    @field_validator("tabs")
    @classmethod
    def _validate_unique_tab_paths(
        cls, value: list[CanvasExtensionPanelTab]
    ) -> list[CanvasExtensionPanelTab]:
        paths = [tab.path for tab in value]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate tab path in a conversation panel")
        return value


class CanvasExtensionContributes(BaseModel):
    """Contributions an extension makes to the Canvas UI."""

    pages: list[CanvasExtensionPage] = Field(
        default_factory=list, description="Pages contributed to Canvas navigation"
    )
    conversation_panels: list[CanvasExtensionConversationPanel] = Field(
        default_factory=list,
        # Omitted when empty, so a manifest without panels dumps as before: a
        # local App's backend approval revision is a hash of that dump.
        exclude_if=lambda value: not value,
        description="Panels opened from buttons in the conversation header",
    )

    @model_validator(mode="after")
    def _validate_unique_contribution_ids(self) -> "CanvasExtensionContributes":
        """Page, panel and tab ids form one namespace per extension."""
        ids = [page.id for page in self.pages]
        for panel in self.conversation_panels:
            ids.append(panel.id)
            ids.extend(tab.id for tab in panel.tabs)
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate contribution id: {duplicates[0]!r}")
        return self

    @field_validator("pages")
    @classmethod
    def _validate_unique_pages(
        cls, v: list[CanvasExtensionPage]
    ) -> list[CanvasExtensionPage]:
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for page in v:
            if page.id in seen_ids:
                raise ValueError(f"Duplicate page contribution id: {page.id!r}")
            if page.path in seen_paths:
                raise ValueError(f"Duplicate page path: {page.path!r}")
            seen_ids.add(page.id)
            seen_paths.add(page.path)
        return v


BackendPlatform = Literal["linux-amd64", "linux-arm64", "darwin-amd64", "darwin-arm64"]


class CanvasExtensionBackendArtifact(BaseModel):
    """Immutable backend artifact for one supported platform."""

    path: str = Field(description="Package-relative .tar.gz artifact path")
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Lowercase SHA-256 checksum"
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if (
            not value
            or value.startswith("/")
            or ".." in Path(value).parts
            or not value.endswith(".tar.gz")
        ):
            raise ValueError("artifact path must be a relative .tar.gz path")
        return value


class CanvasExtensionBackendHealth(BaseModel):
    """Loopback HTTP readiness probe for a backend process."""

    path: str = Field(default="/health", pattern=r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*$")
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    interval_seconds: float = Field(default=0.1, gt=0, le=10)


class CanvasExtensionBackend(BaseModel):
    """Optional trusted backend declaration for schema-1 Canvas Apps."""

    schema_version: Literal[1]
    artifacts: dict[BackendPlatform, CanvasExtensionBackendArtifact] = Field(
        min_length=1
    )
    argv: list[str] = Field(min_length=1)
    health: CanvasExtensionBackendHealth = Field(
        default_factory=CanvasExtensionBackendHealth
    )
    inherit_environment: list[str] = Field(default_factory=list)

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: list[str]) -> list[str]:
        allowed = {"{port}", "{data_dir}", "{artifact_dir}"}
        for argument in value:
            if not argument or "\x00" in argument:
                raise ValueError("backend argv entries must be non-empty")
            placeholders = set(re.findall(r"\{[^{}]+\}", argument))
            if not placeholders.issubset(allowed):
                raise ValueError("backend argv contains an unsupported placeholder")
        return value

    @field_validator("inherit_environment")
    @classmethod
    def _validate_environment(cls, value: list[str]) -> list[str]:
        allowed = {"LANG", "LC_ALL", "LC_CTYPE", "PATH", "TMPDIR", "TZ"}
        if len(value) != len(set(value)):
            raise ValueError("inherit_environment entries must be unique")
        if not set(value).issubset(allowed):
            raise ValueError("inherit_environment contains a disallowed variable")
        return value

    @model_validator(mode="after")
    def _require_artifact_executable(self) -> "CanvasExtensionBackend":
        if "{artifact_dir}" not in self.argv[0]:
            raise ValueError("backend argv executable must be inside {artifact_dir}")
        return self


class CanvasExtensionManifest(BaseModel):
    """Canvas extension manifest (``canvas-extension.json``)."""

    schema_version: int = Field(description="Manifest schema version")
    name: str = Field(description="Extension name (kebab-case)")
    display_name: str = Field(description="Human-readable extension name")
    version: str = Field(description="Extension version")
    description: str = Field(default="", description="Extension description")
    entrypoint: str = Field(
        description=(
            "Path, relative to the extension package root, to the bundle entry file"
        )
    )
    contributes: CanvasExtensionContributes = Field(
        default_factory=CanvasExtensionContributes,
        description="Contributions this extension makes to the Canvas UI",
    )
    backend: CanvasExtensionBackend | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional explicitly prepared and started backend service",
    )

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported canvas extension schema_version")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        validate_extension_name(v)
        return v

    @field_validator("entrypoint")
    @classmethod
    def _validate_entrypoint(cls, v: str) -> str:
        """Reject textual traversal/absolute paths.

        Syntactic only — see :func:`resolve_entrypoint` for the real,
        symlink-aware containment check against the installed package root.
        """
        if not v:
            raise ValueError("entrypoint must not be empty")
        if v.startswith("/"):
            raise ValueError("entrypoint must be relative, not absolute")
        if ".." in Path(v).parts:
            raise ValueError(
                "entrypoint cannot contain '..' (parent directory traversal)"
            )
        return v


def resolve_entrypoint(manifest: CanvasExtensionManifest, package_root: Path) -> Path:
    """Resolve ``manifest.entrypoint`` against ``package_root``, safely.

    Field-level validation on ``entrypoint`` only rejects textual traversal
    (``..``) and absolute paths. It cannot catch a symlink inside the
    package that resolves outside of it. This performs the real
    filesystem-level containment check (resolving symlinks) and must be
    called both when an extension is installed and again immediately
    before its entrypoint is read or served over HTTP.

    Args:
        manifest: A validated manifest.
        package_root: The extension's installed package root directory.

    Returns:
        The resolved, contained entrypoint path.

    Raises:
        ValueError: If the resolved entrypoint escapes ``package_root``, or
            does not resolve to a regular file within it (covers ``.``,
            a directory, a dangling symlink, and symlink cycles — none of
            which ``is_relative_to`` alone rejects).
    """
    return resolve_package_file(package_root, manifest.entrypoint, "entrypoint")


def resolve_package_file(package_root: Path, relative: str, what: str) -> Path:
    """Resolve ``relative`` inside ``package_root`` to a contained regular file.

    Symlinks are resolved before containment is checked.

    Raises:
        ValueError: It escapes the package or is not a regular file.
    """
    root = package_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(
            f"{what} {relative!r} resolves outside the extension package root"
        )
    if not candidate.is_file():
        raise ValueError(
            f"{what} {relative!r} does not resolve to a file in the extension package"
        )
    return candidate


def resolve_panel_icon(
    manifest: CanvasExtensionManifest, panel_id: str, package_root: Path
) -> Path | None:
    """The contained icon file of a panel; None for no such panel or no icon.

    Raises:
        ValueError: The declared icon escapes the package, or does not resolve
            to a regular .svg or .png file.
    """
    for panel in manifest.contributes.conversation_panels:
        if panel.id == panel_id and panel.icon is not None:
            icon = resolve_package_file(package_root, panel.icon, "panel icon")
            if icon.suffix not in PANEL_ICON_MEDIA_TYPES:
                raise ValueError(
                    f"panel icon {panel.icon!r} does not resolve to a .svg or .png file"
                )
            return icon
    return None
