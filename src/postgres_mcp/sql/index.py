import re
from dataclasses import dataclass
from typing import Any


def quote_ident(ident: str) -> str:
    """Quote an identifier (or each part of a dotted name) unless it is an expression or already safe as is."""
    if "(" in ident or " " in ident:
        return ident
    parts = []
    for p in ident.split("."):
        if p.startswith('"') or re.fullmatch(r"[a-z_][a-z0-9_$]*", p):
            parts.append(p)
        else:
            parts.append('"' + p.replace('"', '""') + '"')
    return ".".join(parts)


@dataclass(frozen=True)
class IndexDefinition:
    """Immutable index configuration for hashing."""

    table: str
    columns: tuple[str, ...]
    using: str = "btree"

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "columns": list(self.columns),
            "using": self.using,
            "definition": self.definition,
        }

    @property
    def definition(self) -> str:
        return f"CREATE INDEX {self.name} ON {quote_ident(self.table)} USING {self.using} ({', '.join(quote_ident(c) for c in self.columns)})"

    @property
    def name(self) -> str:
        # Clean column names for use in index naming
        # Replace special characters with underscores to avoid issues with
        # functional expressions
        cleaned_columns = []
        for col in self.columns:
            # Replace parentheses and other special characters with underscores
            # This ensures expressions like LOWER(column_name) work in
            # index names
            cleaned_col = col.replace("(", "_").replace(")", "_").replace(" ", "_").replace(",", "_")
            # Remove consecutive underscores
            while "__" in cleaned_col:
                cleaned_col = cleaned_col.replace("__", "_")
            # Remove trailing underscores
            cleaned_col = cleaned_col.rstrip("_")
            cleaned_columns.append(cleaned_col)

        column_part = "_".join(cleaned_columns)
        suffix = "" if self.using == "btree" else f"_{self.using}"
        base = f"crystaldba_idx_{self.table}_{column_part}_{len(self.columns)}"
        return re.sub(r"[^a-z0-9_]+", "_", f"{base}{suffix}".lower()).strip("_")[:63]

    def __str__(self) -> str:
        return self.definition

    def __repr__(self) -> str:
        return f"IndexConfig(table='{self.table}', columns={self.columns}, using='{self.using}')"
