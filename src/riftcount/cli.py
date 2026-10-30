import os
import sys
from pathlib import Path

import typer

from riftcount._media import media_help
from riftcount.count import count_paths
from riftcount.errors import CountError, UsageError
from riftcount.inventory import dumps_report
from riftcount.weights import default_weights_dir, require_weights_dir, update_weights

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
    help="Inventory photos and videos of cards.",
)


@app.callback()
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(ctx.get_help())
    raise typer.Exit(code=2)


@app.command()
def update(
    release: Path | None = typer.Option(
        None,
        "--release",
        help=f"Weights folder. Defaults to {default_weights_dir()}.",
    ),
) -> None:
    """Download or refresh the cardvision weights into the weights folder."""
    try:
        target = update_weights(release)
        print(f"weights ready at {target}", file=sys.stderr)
    except CountError as err:
        print(str(err), file=sys.stderr)
        raise typer.Exit(code=err.exit_code) from err


@app.command()
def count(
    paths: list[Path] = typer.Argument(..., help=media_help()),
    release: Path | None = typer.Option(
        None,
        "--release",
        help=f"Weights folder. Defaults to {default_weights_dir()}.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Also write the JSON here. The write uses a temp file and replace.",
    ),
) -> None:
    """Inventory each path. Copies are the max times a name appears in one frame."""
    try:
        weights = require_weights_dir(release)
        report = count_paths(paths, release=weights)
        text = dumps_report(report)
        if out is not None:
            _write_out(out, text)
        sys.stdout.write(text)
        for capture in report.captures:
            copies = capture.inventory.estimated_cards
            print(
                f"{capture.path.name} {capture.kind.value} {copies} copies, "
                f"{capture.unnamed_boxes} unnamed, {capture.back_boxes} backs",
                file=sys.stderr,
            )
    except CountError as err:
        print(str(err), file=sys.stderr)
        raise typer.Exit(code=err.exit_code) from err


def _write_out(out: Path, text: str) -> None:
    if out.exists() and out.is_dir():
        raise UsageError(f"output path is a directory: {out}")
    parent = out.parent
    if not parent.is_dir():
        raise UsageError(f"output directory not found: {parent}")
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, out)
