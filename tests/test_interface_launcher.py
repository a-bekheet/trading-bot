from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from trading_bot.interface.launcher import main


class InterfaceLauncherTests(TestCase):
    def test_launches_compiled_fastapi_application_with_explicit_controls(self):
        application = object()
        with (
            patch(
                "trading_bot.interface.launcher.create_app",
                return_value=application,
            ) as create,
            patch("trading_bot.interface.launcher.uvicorn.run") as run,
        ):
            status = main(
                [
                    "--data-dir",
                    "data",
                    "--address",
                    "127.0.0.1",
                    "--port",
                    "9001",
                    "--headless",
                ]
            )

        self.assertEqual(status, 0)
        create.assert_called_once()
        arguments = create.call_args.kwargs
        self.assertEqual(arguments["repo_root"], Path.cwd().resolve())
        self.assertEqual(arguments["data_dir"], (Path.cwd() / "data").resolve())
        self.assertTrue((arguments["static_dir"] / "index.html").is_file())
        run.assert_called_once_with(
            application,
            host="127.0.0.1",
            port=9001,
            reload=False,
            log_level="info",
        )

    def test_launcher_help_exposes_runtime_controls(self):
        with self.assertRaises(SystemExit) as exit_status:
            main(["--help"])

        self.assertEqual(exit_status.exception.code, 0)
