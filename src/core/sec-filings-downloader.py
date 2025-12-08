from typing import List, Optional
from sec_edgar_downloader import Downloader


class SecFilingsDownloader:
    def __init__(self, company_name: str, mail: str, path_to_download: str):
        self.company_name = company_name
        self.mail = mail
        self.path_to_download = path_to_download
        self.downloader = Downloader(
            self.company_name, self.mail, self.path_to_download
        )

    def download_filings(
        self,
        tickers: List[str] | str,
        limit: int,
        before: Optional[str],
        after: Optional[str],
        filing_type: Optional[str] = "10-K",
    ):
        if isinstance(tickers, list):
            for tick in tickers:
                self.downloader.get(filing_type, tick, limit, after, before)
        else:
            self.downloader.get(filing_type, tickers, limit, after, before)
