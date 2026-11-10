"""Keep per-language exports separate when MkDocs builds into a shared site_dir."""

from pathlib import Path

from mkdocs.plugins import event_priority


# The theme and API exporter write to the site root during localized builds.
ROOT_OUTPUTS = ("404.html", "objects.inv")
default_outputs = {}


@event_priority(-50)
def on_post_build(config):
    # Run after the exporters and before i18n starts its nested build (-100).
    i18n = config.plugins["i18n"]
    site = Path(config.site_dir)
    if i18n.is_default_language_build:
        default_outputs.clear()
        for name in ROOT_OUTPUTS:
            path = site / name
            if path.is_file():
                default_outputs[name] = path.read_bytes()
        return

    for name, content in default_outputs.items():
        path = site / name
        path.replace(site / i18n.current_language / name)
        path.write_bytes(content)
