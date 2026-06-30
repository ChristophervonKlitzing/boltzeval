import io
import math
import os
from matplotlib import pyplot as plt
from dataclasses import dataclass

from PIL import Image
import fitz  # PyMuPDF


@dataclass
class PdfBuffer:
    """
    A wrapper to allow differentiation of raw data by type.
    """

    buffer: io.BytesIO

    def __repr__(self):
        size_bytes = len(self.buffer.getbuffer())
        # Format nicely
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024**2:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            size_str = f"{size_bytes / 1024**2:.2f} MB"
        else:
            size_str = f"{size_bytes / 1024**3:.2f} GB"
        return f"PdfBuffer(size={size_str})"


def matplotlib_to_pdf_buffer(obj: plt.Figure | plt.Axes) -> PdfBuffer:
    """
    Convert a matplotlib Figure or Axes to a PDF stored in memory.

    Returns
    -------
    io.BytesIO
        Buffer containing PDF bytes.
    """
    if isinstance(obj, plt.Axes):
        fig = obj.figure
    elif isinstance(obj, plt.Figure):
        fig = obj
    else:
        raise TypeError("Expected matplotlib Figure or Axes")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    buffer.seek(0)
    return PdfBuffer(buffer)


def save_pdf(obj: PdfBuffer, path: str) -> None:
    """
    Save a pdf in the form of a buffer into a PDF file.
    """
    with open(path, "wb") as f:
        f.write(obj.buffer.getbuffer())


def save_pdfs(pdfs: dict[str, PdfBuffer], dirpath: str) -> None:
    """
    Save a dict of pdfs into a directory. dir must exist.
    """
    for name, pdf_buffer in pdfs.items():
        fpath = os.path.join(dirpath, name + ".pdf")
        save_pdf(pdf_buffer, fpath)


def _pdf_bytesio_to_image_PyMuPDF(pdf_bytes: bytes, dpi: int):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

    return img


def pdf_to_pillow_image(pdf_buffer: PdfBuffer, dpi=50):
    # Convert first page to PIL image
    return _pdf_bytesio_to_image_PyMuPDF(pdf_buffer.buffer.getvalue(), dpi=dpi)


def pdf_to_wandb_image(pdf_buffer: PdfBuffer, dpi=50):
    import wandb  # optional dependency (not part of packacke requirements)

    pil_image = pdf_to_pillow_image(pdf_buffer, dpi=dpi)
    return wandb.Image(pil_image)


def plot_pdf(
    pdf_buffer: PdfBuffer,
    dpi=500,
    ax=None,
    show: bool = False,
    title: str | None = None,
    fontsize=16,
):
    """
    This function is mainly for debugging purposes
    """
    img = pdf_to_pillow_image(pdf_buffer, dpi=dpi)

    create_ax = ax is None
    if create_ax:
        width_px, height_px = img.size
        ratio = height_px / width_px
        width = 9
        height = int(math.ceil(ratio * width))
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    ax.imshow(img)
    ax.axis("off")
    # ax.set_title("PDF Preview")
    fig.tight_layout()

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if show:
        plt.show()
    elif create_ax:
        plt.close()
