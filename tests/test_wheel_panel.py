"""Exercise packaged panel assets outside the source checkout, without inference."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


def test_built_wheel_serves_panel_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    built = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(dist), str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stderr
    (wheel,) = dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in ("index.html", "app.js", "memory.js", "style.css"):
            assert archive.read(f"api/ui/{name}") == (root / "ui" / name).read_bytes()
    installed = tmp_path / "installed"
    install = subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(installed), str(wheel)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert install.returncode == 0, install.stderr
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        installed = Path(sys.argv[1])
        sys.path.insert(0, str(installed))
        from fastapi.testclient import TestClient
        from api import server
        from harness.config import Settings
        assert Path(server.__file__).is_relative_to(installed)
        assert server.UI_DIR == installed / 'api' / 'ui'
        class Provider:
            name = 'scripted'
            async def chat(self, *args, **kwargs):
                raise AssertionError('No model call is allowed')
            async def aclose(self):
                pass
        server._build_providers = lambda _: {'scripted': Provider()}
        settings = Settings(_env_file=None, default_provider='scripted',
            enable_support_tools=False,
            memory_db_path=Path('memory.db'), session_db_path=Path('sessions.db'))
        with TestClient(server.create_app(settings=settings)) as client:
            html = client.get('/')
            assert html.status_code == 200 and '/app.js' in html.text
            for name, media in [('app.js', 'javascript'), ('memory.js', 'javascript'),
                                ('style.css', 'css')]:
                result = client.get('/' + name)
                assert result.status_code == 200, name
                assert media in result.headers['content-type'], name
                assert result.content == (server.UI_DIR / name).read_bytes()
            assert client.get('/openapi.json').status_code == 200
            assert client.post('/sessions', json={'user_id': 'wheel-test'}).status_code == 200
        assert 'chromadb' not in sys.modules
    """)
    checked = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
