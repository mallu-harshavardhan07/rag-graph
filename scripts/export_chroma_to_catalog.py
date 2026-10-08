"""Export all indexed CDS View metadata from ChromaDB to data/cds_catalog.json."""

import json
import os
import chromadb


def export_catalog():
    persist_dir = "./data/chroma_db"
    catalog_path = "./data/cds_catalog.json"

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("cds_view_signatures")
    total = collection.count()
    print(f"Total records in ChromaDB: {total:,}")

    chunk_size = 5000
    all_views = []
    seen = set()

    for offset in range(0, total, chunk_size):
        print(f"Fetching chunk offset {offset:,} to {min(offset + chunk_size, total):,}...")
        res = collection.get(include=["metadatas"], limit=chunk_size, offset=offset)
        if res and res.get("metadatas"):
            for m in res["metadatas"]:
                raw = m.get("raw_metadata")
                if raw:
                    try:
                        v = json.loads(raw)
                        vn = v.get("view_name", "").upper()
                        if vn and vn not in seen:
                            seen.add(vn)
                            all_views.append(v)
                    except Exception:
                        pass

    print(f"Extracted {len(all_views):,} unique views from ChromaDB. Saving to {catalog_path}...")
    tmp_path = f"{catalog_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(all_views, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, catalog_path)
    print(f"SUCCESS: Persisted {len(all_views):,} views to {catalog_path} ({os.path.getsize(catalog_path):,} bytes).")


if __name__ == "__main__":
    export_catalog()
