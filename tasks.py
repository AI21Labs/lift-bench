"""
tasks to kick-off the project
"""

from invoke import task


@task
def formatter(tsk, fix=False):
    """
    python format
    """
    auto_fix = "" if fix else "--diff"
    cmd = " && ".join(
        [
            f"ruff format . {auto_fix}",
        ]
    )
    tsk.run(cmd, echo=True, pty=True)


@task
def lint(tsk, fix=False):
    """
    python lint
    """
    auto_fix = "--fix" if fix else ""
    cmd = " && ".join(
        [
            f"ruff check lift_bench/ tests/ {auto_fix}",
        ]
    )
    tsk.run(cmd, echo=True, pty=True)


@task(optional=["coverage"], help={"coverage": "[true|false]"})
def test(tsk, coverage=False):
    """
    Run unit tests
    """
    cov = "--cov --cov-report=term-missing" if coverage else ""
    cmd = f"uv run pytest {cov}"
    tsk.run(cmd, echo=True, pty=True)


@task
def staticcheck(tsk):
    """
    Run static check on the projects files
    """
    cmd = "mypy lift_bench tests"
    tsk.run(cmd, echo=True, pty=True)


@task
def unused_deps(tsk):
    """
    Check for unused dependencies
    """
    cmd = "deptry lift_bench tests scripts"
    tsk.run(cmd, echo=True, pty=True)


@task
def build(tsk):
    """
    generate a package for lift_bench
    """
    cmd = "uv build"
    tsk.run(cmd, echo=True, pty=True)


@task
def update(tsk):
    """
    update outdated packages
    """
    cmd = "uv lock --upgrade"
    tsk.run(cmd, echo=True, pty=True)


@task
def outdated(tsk):
    """
    update outdated packages
    """
    cmd = "uv tree --outdated --depth 1"
    tsk.run(cmd, echo=True, pty=True)
