from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trading_bot.training.manifest import EnvManifest


class ManifestTests(TestCase):
    def test_portfolio_valuation_is_fingerprinted(self):
        liquidation = EnvManifest(portfolio_valuation="liquidation")
        midpoint = EnvManifest(portfolio_valuation="midpoint")

        self.assertNotEqual(liquidation.fingerprint, midpoint.fingerprint)

    def test_quote_freshness_guard_is_fingerprinted(self):
        strict = EnvManifest(max_underlying_quote_age_seconds=300)
        lenient = EnvManifest(max_underlying_quote_age_seconds=1_200)

        self.assertNotEqual(strict.fingerprint, lenient.fingerprint)

    def test_symbol_manifest_ignores_unrelated_ticker_files(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            aapl = data_dir / "AAPL.csv"
            msft = data_dir / "MSFT.csv"
            aapl.write_text("aapl-v1", encoding="utf-8")
            msft.write_text("msft-v1", encoding="utf-8")
            first = EnvManifest.for_directory(data_dir, symbol="AAPL")

            msft.write_text("msft-v2", encoding="utf-8")
            unrelated_change = EnvManifest.for_directory(data_dir, symbol="AAPL")
            aapl.write_text("aapl-v2", encoding="utf-8")
            selected_change = EnvManifest.for_directory(data_dir, symbol="AAPL")

        self.assertEqual(first.data_hash, unrelated_change.data_hash)
        self.assertNotEqual(first.data_hash, selected_change.data_hash)
