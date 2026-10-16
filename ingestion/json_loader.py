import json
from pathlib import Path


class LegalJSONLoader:
    """
    Loads structured Indian legal Acts stored as JSON.

    Expected JSON structure:
    {
        "act_id": "...",
        "act_name": "...",
        "section": "...",
        "title": "...",
        "text": "...",
        "chapter": "...",
        "subsection": "...",
        "effective_date": "...",
        "replaces": "...",
        "punishment_summary": "...",
        "tags": [...],
        "source_url": "..."
    }
    """

    REQUIRED_FIELDS = [
        "act_id",
        "act_name",
        "section",
        "title",
        "text",
    ]

    def __init__(self, json_path):
        self.json_path = Path(json_path)

    def load(self):
        """Load and validate the JSON file."""

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {self.json_path}"
            )

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The file may contain a list directly
        if isinstance(data, list):
            records = data

        # Or the list may be stored under a common key
        elif isinstance(data, dict):
            if "sections" in data:
                records = data["sections"]
            elif "data" in data:
                records = data["data"]
            else:
                raise ValueError(
                    "JSON object does not contain 'sections' or 'data'."
                )

        else:
            raise ValueError(
                "Unsupported JSON structure. Expected list or dictionary."
            )

        if not isinstance(records, list):
            raise ValueError(
                "The sections/data field must contain a list."
            )

        return records

    def validate(self, records):
        """Validate records and report missing/invalid data."""

        valid_records = []
        errors = []

        for index, record in enumerate(records):

            if not isinstance(record, dict):
                errors.append(
                    f"Record {index}: expected dictionary, "
                    f"got {type(record).__name__}"
                )
                continue

            missing_fields = [
                field
                for field in self.REQUIRED_FIELDS
                if field not in record
            ]

            if missing_fields:
                errors.append(
                    f"Record {index}, Section "
                    f"{record.get('section', 'UNKNOWN')}: "
                    f"missing {missing_fields}"
                )
                continue

            section = str(record["section"]).strip()
            title = str(record["title"]).strip()
            text = str(record["text"]).strip()

            if not section:
                errors.append(
                    f"Record {index}: empty section number"
                )
                continue

            if not title:
                errors.append(
                    f"Section {section}: empty title"
                )
                continue

            if not text:
                errors.append(
                    f"Section {section}: empty text"
                )
                continue

            valid_records.append(record)

        return valid_records, errors

    def normalize(self, records):
        """
        Normalize records into the format used by the RAG pipeline.
        """

        normalized = []

        for record in records:

            section = str(record.get("section", "")).strip()

            normalized_record = {
                "act_id": record.get("act_id", ""),
                "act_name": record.get(
                    "act_name",
                    "Unknown Act"
                ),
                "section": section,
                "title": record.get("title", "").strip(),
                "text": record.get("text", "").strip(),
                "chapter": record.get("chapter"),
                "subsection": record.get("subsection"),
                "effective_date": record.get("effective_date"),
                "replaces": record.get("replaces"),
                "punishment_summary": record.get(
                    "punishment_summary"
                ),
                "tags": record.get("tags", []),
                "source_url": record.get("source_url"),
            }

            normalized.append(normalized_record)

        return normalized

    def load_and_prepare(self):
        """
        Complete pipeline:

        JSON
          ↓
        Load
          ↓
        Validate
          ↓
        Normalize
        """

        records = self.load()

        valid_records, errors = self.validate(records)

        normalized_records = self.normalize(valid_records)

        return normalized_records, errors