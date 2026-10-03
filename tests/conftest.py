"""Generate small PDF payloads in memory for real parser tests."""

from collections.abc import Callable
from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


@pytest.fixture
def pdf_factory() -> Callable[..., bytes]:
    def make_pdf(*pages: str | None, encrypted: bool = False) -> bytes:
        writer = PdfWriter()
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        for text in pages:
            page = writer.add_blank_page(width=612, height=792)
            if text is None:
                continue
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
        if encrypted:
            writer.encrypt("test-password")
        output = BytesIO()
        writer.write(output)
        return output.getvalue()

    return make_pdf
