"""
Read an uploaded dataset .zip (ImageFolder layout: top-level folders are
class names, images inside) into in-memory samples, and write back a
cleaned .zip with flagged samples removed.

The sample "id" is the file's path inside the zip — stable and readable,
so a buyer can match a flagged_sample_id in the report straight back to a
file in their own upload.
"""
import io
import os
import zipfile

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetError(ValueError):
    pass


def load_dataset_from_zip(zip_bytes):
    samples = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise DatasetError("Uploaded file is not a valid .zip archive") from exc

    with zf:
        for name in zf.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            raw_bytes = zf.read(name)
            try:
                image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception:
                continue
            samples.append({"id": name, "image": image, "raw_bytes": raw_bytes})

    if not samples:
        raise DatasetError("No readable images found in the uploaded dataset")
    return samples


def write_cleaned_zip(samples, flagged_ids, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample in samples:
            if sample["id"] in flagged_ids:
                continue
            zf.writestr(sample["id"], sample["raw_bytes"])
    return output_path
