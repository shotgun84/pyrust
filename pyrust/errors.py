from __future__ import annotations


class PyRustError(Exception):
    """Diagnostic exception with file, line, column, and source snippet support."""

    def __init__(
        self,
        message: str,
        lineno: int | None = None,
        col: int | None = None,
        filename: str | None = None,
        source_line: str | None = None,
    ) -> None:
        self.message = message
        self.lineno = lineno
        self.col = col
        self.filename = filename or "<string>"
        self.source_line = source_line
        super().__init__(self._format())

    def _format(self) -> str:
        loc = self.filename
        if self.lineno is not None:
            loc += f":{self.lineno}"
            if self.col is not None:
                loc += f":{self.col}"
        header = f"{loc}: PyRustError: {self.message}"
        if self.source_line:
            caret = ""
            if self.col is not None and self.col >= 0:
                caret = "\n  " + (" " * self.col) + "^"
            return f"{header}\n  {self.source_line.rstrip()}{caret}"
        return header
