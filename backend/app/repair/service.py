"""Local, commit-pinned code repair workspaces and protected test execution."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from app.core.errors import DomainError
from app.core.models import CodeRepairTask, uid
from app.core.schemas import CodeRepairPatch

MAX_FILE_CHARS = 20000
MAX_CONTEXT_CHARS = 80000
MAX_TEST_SECONDS = 300
MAX_TEST_OUTPUT = 200000
MAX_TEST_MEMORY = "512m"
MAX_TEST_CPUS = "1"
MAX_TEST_PIDS = "64"
SOURCE_SUFFIXES = {".py"}
SENSITIVE_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "credentials",
    "credentials.json", "secrets.json", "service-account.json", "application_default_credentials.json",
}
SENSITIVE_DIRS = {".ssh", ".aws", ".azure", ".kube", ".docker"}
PROTECTED_NAMES = {
    "pyproject.toml", "pytest.ini", "tox.ini", "noxfile.py", "setup.cfg",
    "setup.py", "conftest.py", ".coveragerc", "requirements.txt", "uv.lock",
    "poetry.lock", "pipfile", "pipfile.lock", "environment.yml", "makefile",
    "justfile", "taskfile.yml", "package.json", "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml", ".drone.yml",
}


def git(cwd: Path, *args: str, input_text: str | None = None, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            input=input_text.encode("utf-8") if input_text is not None else None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise DomainError("Git 操作失败或超时", 400, "git_error") from None
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()[:1000] or "Git 操作失败"
        raise DomainError(message, 400, "git_error")
    return result.stdout.decode("utf-8", errors="replace")


def _allowed_root(settings) -> Path:
    root = settings.code_repository_root
    if root is None:
        raise DomainError("尚未配置 CODE_REPOSITORY_ROOT", 503, "code_repair_disabled")
    try:
        return Path(root).resolve(strict=True)
    except OSError:
        raise DomainError("配置的代码仓库根目录不存在", 503, "code_repair_disabled") from None


def _task_workspaces(settings) -> Path:
    root = Path(settings.code_task_workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_for_task(task: CodeRepairTask) -> Path:
    return Path(task.repository_path)


def _workspace_for_task(task: CodeRepairTask, settings) -> Path:
    base = _task_workspaces(settings)
    try:
        workspace = Path(task.workspace_path).resolve(strict=True)
        workspace.relative_to(base)
    except (OSError, ValueError):
        raise DomainError("任务工作副本路径无效", 409, "workspace_invalid") from None
    if workspace == base or workspace.name != task.id or workspace.parent.name != task.owner_id:
        raise DomainError("任务工作副本身份不匹配", 409, "workspace_invalid")
    return workspace


def _validate_repo(settings, relative_path: str) -> tuple[Path, str, str]:
    root = _allowed_root(settings)
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise DomainError("仓库路径必须相对于 CODE_REPOSITORY_ROOT", 400, "repository_path")
    repo = (root / supplied).resolve()
    try:
        repo.relative_to(root)
    except ValueError:
        raise DomainError("仓库路径超出允许根目录", 403, "repository_path") from None
    if not repo.is_dir():
        raise DomainError("代码仓库目录不存在", 404, "repository_not_found")
    top = Path(git(repo, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != repo:
        raise DomainError("请填写 Git 仓库根目录", 400, "repository_path")
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise DomainError("源仓库有未提交或未跟踪文件；请先提交或清理后再创建任务", 409, "repository_dirty")
    return repo, root.as_posix(), top.as_posix()


def _resolve_commit(repo: Path, base_ref: str) -> str:
    if "\x00" in base_ref or base_ref.startswith("-"):
        raise DomainError("基线引用不合法", 400, "base_ref")
    commit = git(repo, "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise DomainError("基线没有解析为 Git commit", 400, "base_ref")
    return commit


def create_task(settings, owner_id: str, repository_path: str, base_ref: str, objective: str) -> CodeRepairTask:
    repo, _, _ = _validate_repo(settings, repository_path)
    commit = _resolve_commit(repo, base_ref)
    root = _task_workspaces(settings)
    task_id = uid()
    workspace = root / owner_id / task_id
    if workspace == repo or workspace in repo.parents or repo in workspace.parents:
        raise DomainError("任务副本目录必须位于源仓库之外", 400, "workspace_location")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    try:
        git(repo, "worktree", "add", "--detach", str(workspace), commit)
    except Exception:
        raise
    return CodeRepairTask(
        id=task_id,
        owner_id=owner_id,
        repository_path=str(repo),
        workspace_path=str(workspace),
        objective=objective.strip(),
        base_ref=base_ref,
        base_commit=commit,
        selected_files=[],
        status="ready",
    )


def tracked_source_files(task: CodeRepairTask, settings) -> list[str]:
    workspace = _workspace_for_task(task, settings)
    names = git(workspace, "ls-files", "-z")
    return sorted({
        name for name in names.split("\0")
        if name and PurePosixPath(name).suffix.lower() in SOURCE_SUFFIXES
        and not _protected(name)
    })


def _safe_repo_path(path: str) -> str:
    if not path or "\\" in path or "\x00" in path or "\n" in path or "\r" in path:
        raise DomainError("文件路径不合法", 400, "patch_path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise DomainError("文件路径不能离开任务工作副本", 400, "patch_path")
    return pure.as_posix()


def _protected(path: str) -> bool:
    pure = PurePosixPath(path)
    parts = [part.lower() for part in pure.parts]
    name = parts[-1]
    return (
        any(part in {"test", "tests", "testing"} for part in parts)
        or name == "test.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
        or (name.startswith("requirements") and name.endswith(".txt"))
        or name in PROTECTED_NAMES
        or (len(parts) >= 2 and parts[0] == ".github" and parts[1] == "workflows")
        or (len(parts) >= 2 and parts[0] in {".circleci", ".buildkite"})
    )


def read_context(task: CodeRepairTask, settings, selected: list[str]) -> list[dict]:
    available = set(tracked_source_files(task, settings))
    if len(set(selected)) != len(selected):
        raise DomainError("选择的文件不能重复", 400, "file_selection")
    total = 0
    output = []
    for raw_path in selected:
        path = _safe_repo_path(raw_path)
        if path not in available:
            raise DomainError("只能选择任务基线中的 Python 源文件", 400, "file_selection")
        content = git(_workspace_for_task(task, settings), "show",
                      f"{task.base_commit}:{path}")
        if "\x00" in content or len(content) > MAX_FILE_CHARS:
            raise DomainError("所选文件为二进制或超过上下文上限", 400, "file_content")
        total += len(content)
        if total > MAX_CONTEXT_CHARS:
            raise DomainError("选择的文件总量超过上下文上限", 400, "file_content")
        output.append({"path": path, "content": content})
    return output


def _validate_patch(patch: str, selected: list[str]) -> list[str]:
    selected_paths = {_safe_repo_path(path) for path in selected}
    if len(patch) > 30000:
        raise DomainError("补丁超过大小上限", 400, "patch_too_large")
    paths = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            matched = next((path for path in selected_paths
                            if line == f"diff --git a/{path} b/{path}"), None)
            if matched is None:
                raise DomainError("补丁只能修改本次选择的文件，且不能重命名文件", 400, "patch_scope")
            if _protected(matched):
                raise DomainError("测试、测试配置和验收文件不可修改", 403, "protected_file")
            paths.append(matched)
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            raise DomainError("补丁不能重命名或复制文件", 400, "patch_scope")
        elif line.startswith("--- ") or line.startswith("+++ "):
            marker, value = line[:4], line[4:]
            if value == "/dev/null":
                continue
            prefix = "a/" if marker == "--- " else "b/"
            matched = next((path for path in selected_paths if value == prefix + path), None)
            if matched is None:
                raise DomainError("补丁文件头不匹配已选择文件", 400, "patch_scope")
    if not paths or len(paths) != len(set(paths)):
        raise DomainError("补丁为空或包含重复文件段", 400, "patch_format")
    return paths


def apply_patch(task: CodeRepairTask, settings, selected: list[str], patch: str) -> str:
    workspace = _workspace_for_task(task, settings)
    paths = _validate_patch(patch, selected)
    try:
        git(workspace, "apply", "--check", "--whitespace=error-all", "-", input_text=patch)
        git(workspace, "apply", "--whitespace=error-all", "-", input_text=patch)
        git(workspace, "add", "--", *paths)
        if git(workspace, "diff", "--exit-code").strip():
            raise DomainError("补丁应用后工作区含未暂存变更", 409, "workspace_changed")
        actual_paths = set(git(workspace, "diff", "--cached", "--name-only", task.base_commit).splitlines())
        if actual_paths != set(paths) or any(_protected(path) for path in actual_paths):
            raise DomainError("补丁修改范围校验失败", 403, "protected_file")
        return artifact_hash(task, settings)
    except Exception:
        restore_checkpoint(task, settings)
        raise


def artifact_hash(task: CodeRepairTask, settings) -> str:
    workspace = _workspace_for_task(task, settings)
    head = git(workspace, "rev-parse", "HEAD").strip()
    if head != task.base_commit:
        raise DomainError("任务副本基线已改变", 409, "artifact_changed")
    unstaged = git(workspace, "diff", "--name-only").strip()
    if unstaged:
        raise DomainError("任务副本存在未暂存的额外修改", 409, "artifact_changed")
    patch = git(workspace, "diff", "--cached", "--binary", task.base_commit)
    if not patch:
        raise DomainError("任务副本没有可审核的补丁", 409, "artifact_empty")
    paths = git(workspace, "diff", "--cached", "--name-only", task.base_commit).splitlines()
    if any(_protected(path) for path in paths):
        raise DomainError("验收文件在任务副本中被修改", 403, "protected_file")
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def _test_files(task: CodeRepairTask, settings, destination: Path) -> list[str]:
    workspace = _workspace_for_task(task, settings)
    tracked = git(workspace, "ls-files", "-z").split("\0")
    copied = []
    for raw in tracked:
        if not raw:
            continue
        relative = _safe_repo_path(raw)
        path = PurePosixPath(relative)
        if any(part.lower() in {".git", ".venv", "venv", "node_modules", "__pycache__"}
               for part in path.parts):
            continue
        if any(part.lower() in SENSITIVE_DIRS for part in path.parts):
            continue
        name = path.name.lower()
        if name in SENSITIVE_NAMES or name.startswith(".env.") or path.suffix.lower() in {
                ".pem", ".key", ".p12", ".pfx"}:
            continue
        source = workspace.joinpath(*path.parts)
        resolved = source.resolve(strict=True)
        try:
            resolved.relative_to(workspace)
        except ValueError:
            raise DomainError("测试输入包含指向任务副本外部的符号链接", 400, "unsafe_test_input") from None
        if source.is_symlink() or not resolved.is_file():
            raise DomainError("测试输入只能包含普通文件", 400, "unsafe_test_input")
        target = destination.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative)
    return copied


def _docker_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False,
    )
    if result.returncode:
        raise DomainError("Docker 测试镜像不可用；请准备包含目标项目依赖和 pytest 的可信镜像",
                          503, "test_image_unavailable")
    image_id = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise DomainError("Docker 未返回固定镜像摘要", 503, "test_image_unavailable")
    return image_id


def _docker_engine_identity() -> dict:
    endpoint = subprocess.run(
        ["docker", "context", "inspect", "--format", '{{(index .Endpoints "docker").Host}}'],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False,
    )
    host = endpoint.stdout.strip()
    if endpoint.returncode or not host.startswith(("npipe://", "unix://")):
        raise DomainError("代码修复只允许本机 Docker Engine；远程 context 已拒绝", 503,
                          "test_engine_unavailable")
    versions = subprocess.run(
        ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False,
    )
    parts = versions.stdout.strip().split("|")
    if versions.returncode or len(parts) != 2 or not all(parts) or "<no value>" in parts:
        raise DomainError("无法确认本机 Docker Client/Engine 版本", 503, "test_engine_unavailable")
    context = subprocess.run(
        ["docker", "context", "show"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=15, check=False,
    )
    return {"endpoint": host, "context": context.stdout.strip() or "unknown",
            "client_version": parts[0], "server_version": parts[1]}


def _test_environment(image_id: str, files: list[str], engine: dict) -> dict:
    return {
        "image_id": image_id,
        "docker_engine": engine,
        "command": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "network": "none",
        "read_only_root": True,
        "workspace_read_only": True,
        "memory": MAX_TEST_MEMORY,
        "cpus": MAX_TEST_CPUS,
        "pids": MAX_TEST_PIDS,
        "file_manifest_sha256": hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest(),
    }


def verification_hash(task: CodeRepairTask) -> str:
    if not task.test_environment or task.test_exit_code != 0 or not task.artifact_sha256:
        raise DomainError("缺少通过的测试证明，不能生成审核摘要", 409, "verification_missing")
    bundle = {
        "base_commit": task.base_commit,
        "objective": task.objective,
        "selected_files": task.selected_files or [],
        "summary": task.proposed_summary,
        "patch_sha256": task.artifact_sha256,
        "command": task.test_environment["command"],
        "environment": task.test_environment,
        "exit_code": task.test_exit_code,
        "output_sha256": hashlib.sha256((task.test_output or "").encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(bundle, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def run_tests(task: CodeRepairTask, settings) -> tuple[int, str, dict | None]:
    _workspace_for_task(task, settings)
    container = f"paperqa-code-test-{task.id[:24]}"
    output = ""
    code = 127
    environment = None
    try:
        engine = _docker_engine_identity()
        image_id = _docker_image_id(settings.code_test_docker_image)
        with tempfile.TemporaryDirectory(prefix="paperqa-code-test-") as staging:
            staging_path = Path(staging)
            files = _test_files(task, settings, staging_path)
            environment = _test_environment(image_id, files, engine)
            command = [
                "docker", "run", "--rm", "--pull=never", "--name", container,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", f"--pids-limit={MAX_TEST_PIDS}",
                f"--memory={MAX_TEST_MEMORY}", f"--cpus={MAX_TEST_CPUS}",
                "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m",
                "--mount", f"type=bind,source={staging_path},target=/workspace,readonly",
                "--workdir=/workspace", "--user=65532:65532", "--env=HOME=/tmp",
                "--env=PYTHONUTF8=1", "--env=PYTHONDONTWRITEBYTECODE=1", image_id,
                *environment["command"],
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=MAX_TEST_SECONDS, check=False,
            )
            code = result.returncode
            output = (result.stdout + result.stderr)[-MAX_TEST_OUTPUT:]
    except subprocess.TimeoutExpired as exc:
        code = 124
        parts = [part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part or ""
                 for part in (exc.stdout, exc.stderr)]
        output = (f"Docker 中的 pytest 超过 {MAX_TEST_SECONDS} 秒上限，正在清理容器。\n" +
                  "".join(parts)[-MAX_TEST_OUTPUT:])
    except (OSError, DomainError, subprocess.SubprocessError) as exc:
        code, output = 127, str(exc)[:MAX_TEST_OUTPUT]
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True,
                           timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        current_hash = artifact_hash(task, settings)
        if current_hash != task.artifact_sha256:
            raise DomainError("测试运行期间产物发生变化", 409, "artifact_changed")
    except DomainError as exc:
        code = code or 1
        output = f"{output}\n{exc}"[-MAX_TEST_OUTPUT:]
    return code, output, environment


def restore_checkpoint(task: CodeRepairTask, settings) -> None:
    workspace = _workspace_for_task(task, settings)
    repo = _repo_for_task(task)
    registered = git(repo, "worktree", "list", "--porcelain").splitlines()
    if not any(line.startswith("worktree ") and
               os.path.normcase(str(Path(line[9:]).resolve())) == os.path.normcase(str(workspace))
               for line in registered):
        raise DomainError("任务副本已不再注册为源仓库的 worktree", 409, "workspace_invalid")
    git(workspace, "reset", "--hard", task.base_commit)
    git(workspace, "clean", "-fdx")


def propose(task: CodeRepairTask, settings, selected: list[str], provider) -> tuple[str, str, str]:
    context = read_context(task, settings, selected)
    prompt = (
        "只根据所选源文件实现任务。输出可应用的标准 unified diff，头部使用 "
        "diff --git a/路径 b/路径；不得更改测试或配置。本系统会在应用前再次检查路径，并独立运行 pytest。"
    )
    result, _usage = provider.structured(
        "CodeRepair", prompt,
        {"objective": task.objective, "selected_files": context},
        CodeRepairPatch,
    )
    patch = result.diff.strip() + "\n"
    fingerprint = apply_patch(task, settings, selected, patch)
    return result.summary.strip(), patch, fingerprint
