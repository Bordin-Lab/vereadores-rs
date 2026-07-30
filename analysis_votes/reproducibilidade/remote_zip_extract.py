#!/usr/bin/env python3
"""List or extract selected members from a remote, range-enabled ZIP file.

This utility avoids downloading the complete multi-gigabyte TSE archives.  It
reads the ZIP central directory with HTTP Range requests and retrieves only the
compressed bytes of explicitly selected members.
"""

from __future__ import annotations

import argparse
import binascii
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path


EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


def request_range(url: str, start: int | None, end: int | None) -> bytes:
    if start is None:
        range_value = f"bytes=-{end}"
    else:
        range_value = f"bytes={start}-{end}"
    request = urllib.request.Request(
        url,
        headers={
            "Range": range_value,
            "User-Agent": "TSE-research-reproducibility/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise RuntimeError(
                f"Servidor não respeitou Range ({range_value}); status={status}"
            )
        return response.read()


def content_length(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "TSE-research-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return int(response.headers["Content-Length"])


def central_directory(url: str) -> tuple[int, list[dict[str, int | str]]]:
    total_size = content_length(url)
    tail_size = min(total_size, 1024 * 1024)
    tail = request_range(url, None, tail_size)
    eocd_index = tail.rfind(EOCD_SIGNATURE)
    if eocd_index < 0:
        raise RuntimeError("EOCD não encontrado no fim do arquivo ZIP")
    eocd = tail[eocd_index : eocd_index + 22]
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack("<4s4H2LH", eocd)
    if signature != EOCD_SIGNATURE:
        raise RuntimeError("Assinatura EOCD inválida")
    if disk_number or central_disk or disk_entries != total_entries:
        raise RuntimeError("ZIP multidisco não suportado")
    if 0xFFFF in (disk_entries, total_entries) or 0xFFFFFFFF in (
        central_size,
        central_offset,
    ):
        raise RuntimeError("ZIP64 não suportado por este utilitário")
    central = request_range(
        url, central_offset, central_offset + central_size - 1
    )
    entries: list[dict[str, int | str]] = []
    position = 0
    while position < len(central):
        fixed = central[position : position + 46]
        if len(fixed) < 46:
            raise RuntimeError("Cabeçalho central truncado")
        fields = struct.unpack("<4s6H3L5H2L", fixed)
        if fields[0] != CENTRAL_SIGNATURE:
            raise RuntimeError(
                f"Assinatura central inválida na posição {position}"
            )
        (
            _,
            _version_made,
            _version_needed,
            flag,
            method,
            _mod_time,
            _mod_date,
            crc32_value,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attributes,
            _external_attributes,
            local_offset,
        ) = fields
        start_name = position + 46
        raw_name = central[start_name : start_name + name_length]
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = raw_name.decode(encoding)
        entries.append(
            {
                "name": name,
                "flag": flag,
                "method": method,
                "crc32": crc32_value,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
        )
        position += 46 + name_length + extra_length + comment_length
    if len(entries) != total_entries:
        raise RuntimeError(
            f"Diretório central contém {len(entries)} entradas; "
            f"EOCD informa {total_entries}"
        )
    return total_size, entries


def extract_member(url: str, entry: dict[str, int | str], output: Path) -> None:
    local_offset = int(entry["local_offset"])
    fixed = request_range(url, local_offset, local_offset + 29)
    (
        signature,
        _version,
        flag,
        method,
        _mod_time,
        _mod_date,
        _crc32_local,
        _compressed_local,
        _uncompressed_local,
        name_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", fixed)
    if signature != LOCAL_SIGNATURE:
        raise RuntimeError(f"Cabeçalho local inválido para {entry['name']}")
    if flag & 0x1:
        raise RuntimeError(f"Membro criptografado não suportado: {entry['name']}")
    if method not in (0, 8):
        raise RuntimeError(
            f"Método de compressão {method} não suportado: {entry['name']}"
        )
    data_offset = local_offset + 30 + name_length + extra_length
    compressed_size = int(entry["compressed_size"])
    compressed = request_range(
        url, data_offset, data_offset + compressed_size - 1
    )
    if len(compressed) != compressed_size:
        raise RuntimeError(
            f"Download incompleto de {entry['name']}: "
            f"{len(compressed)} != {compressed_size}"
        )
    if method == 0:
        payload = compressed
    else:
        payload = zlib.decompress(compressed, -15)
    expected_size = int(entry["uncompressed_size"])
    if len(payload) != expected_size:
        raise RuntimeError(
            f"Tamanho inválido de {entry['name']}: "
            f"{len(payload)} != {expected_size}"
        )
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    if crc != int(entry["crc32"]):
        raise RuntimeError(
            f"CRC inválido de {entry['name']}: "
            f"{crc:08x} != {int(entry['crc32']):08x}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--match", required=True, help="Expressão regular")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--list-only", action="store_true", help="Somente listar membros"
    )
    args = parser.parse_args()

    total_size, entries = central_directory(args.url)
    pattern = re.compile(args.match, re.IGNORECASE)
    selected = [entry for entry in entries if pattern.search(str(entry["name"]))]
    print(f"arquivo_remoto_bytes={total_size}", file=sys.stderr)
    print(f"membros_zip={len(entries)}", file=sys.stderr)
    print(f"membros_selecionados={len(selected)}", file=sys.stderr)
    for entry in selected:
        print(
            f"{entry['name']}\t{entry['compressed_size']}\t"
            f"{entry['uncompressed_size']}"
        )
    if args.list_only:
        return
    if not args.output_dir:
        parser.error("--output-dir é obrigatório para extração")
    for entry in selected:
        target = args.output_dir / Path(str(entry["name"])).name
        print(f"extraindo {entry['name']} -> {target}", file=sys.stderr)
        extract_member(args.url, entry, target)


if __name__ == "__main__":
    main()
