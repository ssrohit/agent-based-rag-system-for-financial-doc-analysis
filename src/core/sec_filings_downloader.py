import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from sec_edgar_downloader import Downloader

from src.utils.timing import log_duration

logger = logging.getLogger(__name__)


class SecFilingsDownloader:
    def __init__(self, company_name: str, mail: str, path_to_download: str):
        self.company_name = company_name
        self.mail = mail
        self.path_to_download = path_to_download

    def _flatten_and_move(self, temp_dir: str) -> List[Path]:
        """Walk the nested temp download directory and move every file to the
        flat ``path_to_download`` folder.

        The ``sec_edgar_downloader`` library creates:
            <temp_dir>/sec-edgar-filings/<TICKER>/<FORM_TYPE>/<ACCESSION>/<filename>

        Each file is moved to:
            <path_to_download>/<TICKER>_<FORM_TYPE>_<ACCESSION>_<filename>

        Returns a list of destination ``Path`` objects for every file moved.
        """
        dest_base = Path(self.path_to_download)
        dest_base.mkdir(parents=True, exist_ok=True)

        sec_edgar_root = Path(temp_dir) / "sec-edgar-filings"
        if not sec_edgar_root.exists():
            logger.warning("Expected 'sec-edgar-filings' dir not found in temp path: %s", temp_dir)
            return []

        moved: List[Path] = []
        for file_path in sec_edgar_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Relative parts after sec-edgar-filings/: TICKER / FORM_TYPE / ACCESSION / filename
            parts = file_path.relative_to(sec_edgar_root).parts
            if len(parts) != 4:
                logger.warning("Unexpected path depth, skipping: %s", file_path)
                continue

            ticker, form_type, accession, filename = parts
            flat_name = f"{ticker}_{form_type}_{accession}_{filename}"
            dest = dest_base / flat_name
            shutil.move(str(file_path), str(dest))
            logger.debug("Moved filing to: %s", dest)
            moved.append(dest)

        return moved

    def download_filings(
        self,
        tickers: List[str] | str,
        limit: int,
        before: Optional[str],
        after: Optional[str],
        filing_type: Optional[str] = "10-K",
    ) -> List[Path]:
        """Download SEC filings to a temporary directory, then flatten and move
        all files to ``path_to_download`` using the naming convention
        ``{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}``.

        Returns a list of ``Path`` objects for every file that was moved.
        """
        temp_dir = tempfile.mkdtemp()
        logger.debug("Downloading filings to temp dir: %s", temp_dir)

        try:
            downloader = Downloader(self.company_name, self.mail, temp_dir)

            after_date = after.split("T")[0] if after is not None else None
            before_date = before.split("T")[0] if before is not None else None

            if isinstance(tickers, list):
                for tick in tickers:
                    with log_duration(logger, f"SEC EDGAR download ({tick}, {filing_type})"):
                        downloader.get(
                            form=filing_type or "10-K",
                            ticker_or_cik=tick,
                            limit=limit,
                            after=after_date,
                            before=before_date,
                        )
            else:
                with log_duration(logger, f"SEC EDGAR download ({tickers}, {filing_type})"):
                    downloader.get(
                        form=filing_type or "10-K",
                        ticker_or_cik=tickers,
                        limit=limit,
                        after=after_date,
                        before=before_date,
                    )

            return self._flatten_and_move(temp_dir)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", temp_dir)
